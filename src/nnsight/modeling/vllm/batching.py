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
    """Batcher that handles tensor-parallel gather/split and inference-mode
    tensor cloning for vLLM.

    **Tensor-parallel (TP) support:**
    vLLM's ``ColumnParallelLinear`` and ``RowParallelLinear`` layers
    shard tensors across GPUs. This batcher transparently gathers the
    sharded tensors so the user sees full (unsharded) values, then
    splits them back before returning control to vLLM.

    **Inference-mode cloning:**
    vLLM runs under ``torch.inference_mode()``, producing tensors that
    reject in-place operations.  ``pre_user_transform`` clones output
    tensors out of inference mode so users can do ``output[0][:] = 0``.
    ``post_user_transform`` clones again before returning to vLLM so
    fused-kernel mutations don't corrupt ``.save()``'d references.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_module = None
        self.parallel = False
        self.gathered = False
        self.type = None

    def wrap(self, model: Envoy):
        """Register TP gather/split hooks when sharding across GPUs."""
        from vllm.distributed.parallel_state import get_tp_group
        if get_tp_group().world_size > 1:
            self._register_tp_hooks(model)

    def _register_tp_hooks(self, model: Envoy):
        """Register tensor-parallel gather/split hooks."""

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

                    output = apply(
                        output,
                        lambda x: split_tensor_along_last_dim(
                            x, num_partitions=self.current_module.tp_size
                        )[self.current_module.tp_rank].contiguous(),
                        torch.Tensor,
                    )

                elif isinstance(self.current_module, RowParallelLinear):

                    output = apply(
                        output, lambda x: x / self.current_module.tp_size, torch.Tensor
                    )

            self.parallel = False
            self.gathered = False
            self.current_module = None
            self.type = None

            return output

        for module in model.modules():
            if isinstance(module._module, (RowParallelLinear, ColumnParallelLinear)):
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

    # ---- Inference-mode and save-protection cloning ----

    def pre_user_transform(self, module: torch.nn.Module, hook_type: str, value: Any, is_skip: bool = False) -> Any:
        """Clone inference-mode output tensors so users can do in-place ops."""
        if hook_type == "output":
            value = self._clone_inference_tensors(value)
        return value

    def post_user_transform(self, module: torch.nn.Module, hook_type: str, value: Any, is_skip: bool = False) -> Any:
        """Clone output tensors before returning to vLLM to protect .save()
        references from downstream in-place mutations by fused kernels."""
        if hook_type == "output":
            value = self._clone_output_tensors(value)
        return value

    @staticmethod
    def _clone_inference_tensors(value: Any) -> Any:
        """Clone inference-mode tensors out of inference mode."""
        if isinstance(value, torch.Tensor) and value.is_inference():
            with torch.inference_mode(False):
                return value.clone()
        elif isinstance(value, tuple):
            cloned = []
            changed = False
            for v in value:
                if isinstance(v, torch.Tensor) and v.is_inference():
                    with torch.inference_mode(False):
                        cloned.append(v.clone())
                    changed = True
                else:
                    cloned.append(v)
            return tuple(cloned) if changed else value
        return value

    @staticmethod
    def _clone_output_tensors(value: Any) -> Any:
        """Clone all tensors in a value to protect against in-place mutation."""
        if isinstance(value, torch.Tensor):
            return value.clone()
        elif isinstance(value, tuple):
            return tuple(
                v.clone() if isinstance(v, torch.Tensor) else v
                for v in value
            )
        return value
