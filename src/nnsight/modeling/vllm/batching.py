from typing import Any, Union
import torch
from ...intervention.batching import Batcher, apply
from ...intervention.envoy import Envoy
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    tensor_model_parallel_all_gather,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)


class VLLMBatcher(Batcher):

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

        for module in model.modules():

            module._module.register_forward_pre_hook(
                pre_input_hook, prepend=True, with_kwargs=True
            )
            module._module.register_forward_pre_hook(
                post_input_hook, prepend=False, with_kwargs=True
            )
            module._module.register_forward_hook(pre_output_hook, prepend=True)
            module._module.register_forward_hook(post_output_hook, prepend=False)

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
