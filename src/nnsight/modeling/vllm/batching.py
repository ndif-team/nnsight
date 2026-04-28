from typing import Any, Union
import torch
from ...intervention.batching import Batcher, apply
from ...intervention.envoy import Envoy
from ...intervention.hooks import add_ordered_hook
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    tensor_model_parallel_all_gather,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)


class VLLMBatcher(Batcher):
    """Batcher that handles tensor-parallel gather/split for vLLM.

    vLLM's ``ColumnParallelLinear`` and ``RowParallelLinear`` layers
    shard tensors across GPUs. When NNsight intervention code accesses
    inputs or outputs of these layers, this batcher transparently
    gathers the sharded tensors so the user sees the full (unsharded)
    values, then splits them back before returning control to vLLM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_module = None
        self.parallel = False
        self.gathered = False
        self.type = None

    def wrap(self, model: Envoy):

        def pre_input_hook(module: torch.nn.Module, args: Any, kwargs: Any):
            self.current_module = module
            self.type = "input"

            if isinstance(module, RowParallelLinear):
                self.parallel = module.input_is_parallel

        def post_input_hook(module: torch.nn.Module, args: Any, kwargs: Any):

            if self.parallel and self.gathered:

                if isinstance(self.current_module, RowParallelLinear):

                    args, kwargs = apply(
                        (args, kwargs),
                        lambda x: split_tensor_along_last_dim(
                            x, num_partitions=self.current_module.tp_size
                        )[self.current_module.tp_rank].contiguous(),
                        torch.Tensor,
                    )

            self.parallel = False
            self.gathered = False
            self.current_module = None
            self.type = None

            return args, kwargs

        def pre_output_hook(module: torch.nn.Module, args: Any, output: Any):

            self.current_module = module
            self.type = "output"

            if isinstance(module, ColumnParallelLinear):
                self.parallel = not module.gather_output
            elif isinstance(module, RowParallelLinear):
                self.parallel = not module.reduce_results

        def post_output_hook(module: torch.nn.Module, args: Any, output: Any):

            if self.parallel and self.gathered:

                if isinstance(self.current_module, ColumnParallelLinear):

                    # Undo tensor_model_parallel_all_gather by splitting back along last dimension
                    output = apply(
                        output,
                        lambda x: split_tensor_along_last_dim(
                            x, num_partitions=self.current_module.tp_size
                        )[self.current_module.tp_rank].contiguous(),
                        torch.Tensor,
                    )

                elif isinstance(self.current_module, RowParallelLinear):

                    # Undo tensor_model_parallel_all_reduce by dividing by tp_size
                    # Since all_reduce sums across ranks, dividing gives each rank 1/tp_size of the sum
                    output = apply(
                        output, lambda x: x / self.current_module.tp_size, torch.Tensor
                    )

            self.parallel = False
            self.gathered = False
            self.current_module = None
            self.type = None

            return output

        # Mark the hooks with mediator_idx so they sort correctly when
        # add_ordered_hook rebuilds the hook dict for intervention hooks:
        #
        # - pre_* hooks use -inf so they fire BEFORE any mediator hook
        #   (they set up the batcher state that narrow/gather depend on).
        # - post_* hooks use +inf so they fire AFTER every mediator hook
        #   (they clean up after all interventions have read/modified
        #   the potentially-gathered values, and split the tensors
        #   back to their sharded form before vLLM resumes).
        pre_input_hook.mediator_idx = float("-inf")
        post_input_hook.mediator_idx = float("inf")
        pre_output_hook.mediator_idx = float("-inf")
        post_output_hook.mediator_idx = float("inf")

        for module in model.modules():
            if isinstance(module._module, (RowParallelLinear, ColumnParallelLinear)):
                add_ordered_hook(module._module, pre_input_hook, "input")
                add_ordered_hook(module._module, post_input_hook, "input")
                add_ordered_hook(module._module, pre_output_hook, "output")
                add_ordered_hook(module._module, post_output_hook, "output")

    def check_gathered(self):

        if self.parallel and not self.gathered:

            if isinstance(self.current_module, ColumnParallelLinear):

                if self.type == "output":

                    self.current_value = apply(
                        self.current_value,
                        lambda x: tensor_model_parallel_all_gather(x),
                        torch.Tensor,
                    )

            elif isinstance(self.current_module, RowParallelLinear):

                if self.type == "input":

                    self.current_value = apply(
                        self.current_value,
                        lambda x: tensor_model_parallel_all_gather(x),
                        torch.Tensor,
                    )

                elif self.type == "output":

                    self.current_value = apply(
                        self.current_value,
                        lambda x: tensor_model_parallel_all_reduce(x),
                        torch.Tensor,
                    )

            self.gathered = True

    def narrow(self, batch_group: Union[int, None]):

        self.check_gathered()

        return super().narrow(batch_group)

    def swap(self, batch_group: Union[int, None], swap_value: Any):

        self.check_gathered()

        return super().swap(batch_group, swap_value)
