import inspect
import logging
from typing import Any, List, Union

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

logger = logging.getLogger(__name__)


"""
Module detection design
-----------------------
The compat layer uses a two-layer defense to avoid false transforms:

1. **Detection** (once at model init via ``_detect_modules``):
   Heuristic checks flag modules as *candidates* for transformation.
   These are intentionally broad — they identify modules that *might*
   need transforms, not modules that definitely do.

   - ``_is_dual_residual_layer``: checks ``'residual' in forward()``
     signature.  Works for all current vLLM decoder layers (Qwen2,
     Llama, Mistral, Mixtral, Gemma2, DeepSeek-V2, Phi3, …).
     GPT2Block has no ``residual`` param → correctly excluded.
     **Limitation**: a module with a ``residual`` parameter used for
     an unrelated purpose would be a false positive.

   - ``_is_vllm_rmsnorm``: ``isinstance(module, vllm...RMSNorm)``.
     **Limitation**: doesn't cover ``LayerNorm`` if a model uses that
     with the same fused ``(normalized, residual)`` return pattern.
     Also, a standalone ``RMSNorm(x)`` call without a ``residual``
     argument returns a plain tensor, not a tuple.

2. **Runtime guard** (every hook call in ``pre/post_user_transform``):
   Before transforming, the code checks the *actual value* format
   (e.g. ``isinstance(value, tuple) and len(value) == 2`` with both
   elements being tensors).  If the value doesn't match the expected
   structure, the transform is silently skipped and the value passes
   through untouched.  A warning is logged on the first mismatch so
   users can diagnose unexpected detection hits.

This means detection false positives are safe — the runtime guard
catches them and falls back to no-op.  Detection false negatives are
the real risk: a module that *should* be transformed but isn't
detected will silently expose raw vLLM semantics.  For now this is
acceptable because all mainstream vLLM architectures are covered.
"""


def _is_dual_residual_layer(module: torch.nn.Module) -> bool:
    """Check if a module uses vLLM's dual-residual-stream pattern.

    Returns True if the module's ``forward()`` accepts a ``residual``
    parameter, which indicates it returns ``(hidden_states, residual)``
    and expects the next layer's fused norm to combine them.

    This is a heuristic — see "Module detection design" above for
    limitations and the runtime guard that catches false positives.
    """
    try:
        sig = inspect.signature(module.__class__.forward)
        return "residual" in sig.parameters
    except (ValueError, TypeError):
        return False


def _is_vllm_rmsnorm(module: torch.nn.Module) -> bool:
    """Check if a module is vLLM's fused RMSNorm (returns tuple).

    Uses isinstance against ``vllm.model_executor.layers.layernorm.RMSNorm``.
    See "Module detection design" above for limitations.
    """
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
        return isinstance(module, RMSNorm)
    except ImportError:
        return False


class VLLMBatcher(Batcher):
    """Batcher that handles tensor-parallel gather/split and HF-compatibility
    transforms for vLLM.

    **Tensor-parallel (TP) support:**
    vLLM's ``ColumnParallelLinear`` and ``RowParallelLinear`` layers
    shard tensors across GPUs. This batcher transparently gathers the
    sharded tensors so the user sees full (unsharded) values, then
    splits them back before returning control to vLLM.

    **HF-compatibility transforms (``compat=True``):**
    vLLM's decoder layers return ``(mlp_output, residual)`` instead of
    HF's ``(hidden_states,)``.  When ``compat`` is enabled, this batcher
    transparently combines the dual streams for the user and decomposes
    them back for vLLM, so users can write the same intervention code
    regardless of backend.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TP state
        self.current_module = None
        self.parallel = False
        self.gathered = False
        self.type = None

        # Compat state
        self.compat = False
        self._decoder_layer_ids: set = set()
        self._rmsnorm_ids: set = set()
        self._row_parallel_ids: set = set()
        self._transform_frames: dict = {}  # (id(module), hook_type) -> frame dict
        self._guard_warned: set = set()  # module ids that already logged a mismatch warning

    def wrap(self, model: Envoy, compat: bool = True):
        """Register hooks for TP gather/split and compat transforms.

        Args:
            model: The NNsight-wrapped vLLM model.
            compat: If True, enable HF-compatibility transforms for
                decoder layers, RMSNorm, and RowParallelLinear outputs.
        """
        self.compat = compat

        if compat:
            self._detect_modules(model)

        # ---- TP hooks (only when sharding across GPUs) ----
        from vllm.distributed.parallel_state import get_tp_group
        if get_tp_group().world_size > 1:
            self._register_tp_hooks(model)

    def _detect_modules(self, model: Envoy):
        """Identify modules that need compat transforms."""
        for envoy in model.modules():
            actual = envoy._module
            mid = id(actual)

            if _is_dual_residual_layer(actual):
                self._decoder_layer_ids.add(mid)

            if _is_vllm_rmsnorm(actual):
                self._rmsnorm_ids.add(mid)

            if isinstance(actual, RowParallelLinear):
                self._row_parallel_ids.add(mid)

        if self._decoder_layer_ids:
            logger.debug(
                "vLLM compat: detected %d decoder layers, %d RMSNorm, %d RowParallel",
                len(self._decoder_layer_ids),
                len(self._rmsnorm_ids),
                len(self._row_parallel_ids),
            )

    def _register_tp_hooks(self, model: Envoy):
        """Register tensor-parallel gather/split hooks (existing logic)."""

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

    # ---- TP narrow/swap (unchanged) ----

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

    # ---- HF-compatibility transforms ----

    def pre_user_transform(self, module: torch.nn.Module, hook_type: str, value: Any, is_skip: bool = False) -> Any:
        if not self.compat:
            return value

        mid = id(module)
        frame = {}

        if hook_type == "output":
            if mid in self._decoder_layer_ids:
                if (isinstance(value, tuple) and len(value) == 2
                        and isinstance(value[0], torch.Tensor)
                        and isinstance(value[1], torch.Tensor)):
                    out0, out1 = value
                    frame["type"] = "decoder_layer"
                    frame["original"] = (out0, out1)
                    # Exit inference_mode to create a normal tensor that
                    # users can modify in-place (e.g. output[0][:] = 0).
                    with torch.inference_mode(False):
                        combined = out0.clone() + out1.clone()
                    value = (combined,)
                elif mid not in self._guard_warned:
                    self._guard_warned.add(mid)
                    logger.warning(
                        "vLLM compat: decoder layer %s detected but output is %s, "
                        "not (Tensor, Tensor) 2-tuple — skipping transform",
                        getattr(module, "__path__", type(module).__name__),
                        type(value).__name__ if not isinstance(value, tuple)
                        else f"{len(value)}-tuple",
                    )

            elif mid in self._rmsnorm_ids:
                if (isinstance(value, tuple) and len(value) == 2
                        and isinstance(value[0], torch.Tensor)
                        and isinstance(value[1], torch.Tensor)):
                    frame["type"] = "rmsnorm"
                    frame["residual"] = value[1]
                    with torch.inference_mode(False):
                        value = value[0].clone()
                elif mid not in self._guard_warned:
                    self._guard_warned.add(mid)
                    logger.warning(
                        "vLLM compat: RMSNorm %s detected but output is %s, "
                        "not (Tensor, Tensor) 2-tuple — skipping transform",
                        getattr(module, "__path__", type(module).__name__),
                        type(value).__name__ if not isinstance(value, tuple)
                        else f"{len(value)}-tuple",
                    )

            elif mid in self._row_parallel_ids:
                if (isinstance(value, tuple) and len(value) == 2
                        and isinstance(value[0], torch.Tensor)):
                    frame["type"] = "row_parallel"
                    frame["bias"] = value[1]
                    with torch.inference_mode(False):
                        value = value[0].clone()
                elif mid not in self._guard_warned:
                    self._guard_warned.add(mid)
                    logger.warning(
                        "vLLM compat: RowParallelLinear %s detected but output is %s, "
                        "not (Tensor, *) 2-tuple — skipping transform",
                        getattr(module, "__path__", type(module).__name__),
                        type(value).__name__ if not isinstance(value, tuple)
                        else f"{len(value)}-tuple",
                    )

        elif hook_type == "input":
            if mid in self._decoder_layer_ids:
                args, kwargs = value
                if len(args) >= 3 and isinstance(args[1], torch.Tensor):
                    positions, hidden, residual = args[0], args[1], args[2]
                    rest = args[3:]
                    frame["type"] = "decoder_layer_input"
                    frame["positions"] = positions
                    frame["original_hidden"] = hidden
                    frame["original_residual"] = residual
                    frame["rest"] = rest
                    with torch.inference_mode(False):
                        if isinstance(residual, torch.Tensor):
                            combined = hidden.clone() + residual.clone()
                        else:
                            # First layer: residual is None, hidden IS the full state
                            combined = hidden.clone()
                    frame["combined_id"] = id(combined)
                    value = (combined,) + rest, kwargs

        # Clone any remaining inference-mode tensors so users can do in-place
        # ops (e.g. output[0][:] = 0).  Dual-stream models already get cloned
        # above; this catches single-stream models (GPT-2, GPT-J, etc.) and
        # any module not covered by the specific transforms.
        if hook_type == "output" and frame.get("type") is None:
            value = self._clone_inference_tensors(value)

        self._transform_frames[(mid, hook_type)] = frame
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

    def post_user_transform(self, module: torch.nn.Module, hook_type: str, value: Any, is_skip: bool = False) -> Any:
        if not self.compat:
            return value

        mid = id(module)

        # Retrieve the frame set by pre_user_transform for this module+hook.
        frame = self._transform_frames.pop((mid, hook_type), {})
        frame_type = frame.get("type")

        if hook_type == "output":

            if is_skip and mid in self._decoder_layer_ids:
                # User provided a skip value in HF format — decompose for vLLM.
                # Put everything in residual stream: fused_add_rms_norm(0, v) = rms_norm(v).
                # Clone to protect user's .save()'d reference from in-place mutation.
                if isinstance(value, tuple) and len(value) == 1:
                    v = value[0]
                elif isinstance(value, torch.Tensor):
                    v = value
                else:
                    return value
                return (torch.zeros_like(v), v.clone())

            if frame_type == "decoder_layer":
                # Decompose user's (possibly modified) value back to dual-stream.
                # Clone before returning to vLLM so fused_add_rms_norm mutates
                # the clone, not the user's .save()'d reference (fixes Gap 1.1).
                if isinstance(value, tuple) and len(value) >= 1:
                    combined = value[0]
                elif isinstance(value, torch.Tensor):
                    combined = value
                else:
                    return value
                return (combined.clone(), torch.zeros_like(combined))

            elif frame_type == "rmsnorm":
                residual = frame["residual"]
                if isinstance(value, torch.Tensor):
                    return (value, residual)

            elif frame_type == "row_parallel":
                bias = frame["bias"]
                if isinstance(value, torch.Tensor):
                    return (value, bias)

        elif hook_type == "input":

            if frame_type == "decoder_layer_input":
                args, kwargs = value
                positions = frame["positions"]
                rest = frame["rest"]

                if args and id(args[0]) == frame.get("combined_id"):
                    # User didn't modify — restore exact originals to avoid fp error.
                    original_hidden = frame["original_hidden"]
                    original_residual = frame["original_residual"]
                    return (positions, original_hidden, original_residual) + rest, kwargs
                else:
                    # User modified — decompose: put everything in hidden.
                    combined = args[0] if args else frame["original_hidden"]
                    original_residual = frame["original_residual"]
                    if isinstance(original_residual, torch.Tensor):
                        new_residual = torch.zeros_like(combined)
                    else:
                        new_residual = None  # First layer: residual stays None
                    return (positions, combined, new_residual) + rest, kwargs

        return value
