"""Attention probabilities, scores, and logsumexp from PyTorch's fused SDPA kernels.

``torch.nn.functional.scaled_dot_product_attention`` dispatches to one of several
fused backends (FlashAttention, memory-efficient attention, cuDNN, or CPU flash
attention) or to an fp32 math fallback. The fused backends never write the
attention probability matrix to global memory, but each one computes the per-row
logsumexp (``lse``) in fp32 for its backward pass before the high-level wrapper
discards it. Using ``lse`` and the fp32 scores ``S``, the attention probabilities
computed by the kernel can be reconstructed as ``exp(S - lse)``, which is the
same formula FlashAttention uses in its backward pass. This reconstruction is
accurate up to fp32 rounding of ``S`` and ``lse`` (rows sum to 1 within a few
ulps of the row's largest ``|S|``: 5e-7 on GPT-2, and 8e-4 on a Qwen2.5-0.5B
layer whose scores reach 1700). By contrast, switching the model to eager
attention in bf16 changes the probabilities by up to 19% at the top key.

When a worker waits on an attention module's kernel location, `serve` runs the
SDPA call inside a `TorchDispatchMode` (`_Capture`) that records the low-level
ATen operator selected by the dispatcher along with its preprocessed arguments
(after mask conversion, stride padding, and GQA expansion) and its return tuple.
The operator runs with the exact arguments the model passed, leaving the model's
output unchanged.

Two backends (memory-efficient attention and cuDNN) only populate the logsumexp
buffer when their ``compute_log_sumexp`` flag is set. When a worker requests
``lse`` or ``probs`` on one of those backends and the flag was ``False`` during the
forward pass, the operator is called a second time under ``torch.no_grad()`` with
``compute_log_sumexp=True``. Its attention output is checked for bitwise equality
against the first launch; if the two outputs differ (which can happen on the
memory-efficient kernel under ``no_grad`` with a stride-0 key head and an
additive bias at certain sequence lengths), the second launch is discarded,
``lse`` is recomputed in fp32 from the scores, and the provenance kind is marked
as ``"replay"``. When no fused backend fires, the fp32 math fallback ran, so
``lse`` is computed in fp32 and also marked as ``"replay"``.

Available sites (each accepting ``heads=``, ``query=``, and, for ``scores`` and
``probs``, ``key=`` as an int, slice, list, or tensor of indices with PyTorch
indexing semantics):

* ``lse``: per-row logsumexp normalizer of shape ``[batch, heads, queries]``,
  served as an ``"exact export"``.
* ``scores``: ``q . k * scale`` in fp32 plus the attention mask (``-inf`` for
  masked positions under a causal or boolean mask, or the float mask values
  directly), of shape ``[batch, heads, queries, keys]``.
* ``probs``: ``exp(scores - lse)``, of shape ``[batch, heads, queries, keys]``.
  Masked positions are ``0``, rows where every key is masked (such as padding
  queries) are all ``0``, and all other rows sum to 1.

.. code-block:: python

    with model.trace(prompt):
        p = model.model.layers[6].self_attn.kernel.probs(heads=4, query=-1).save()

Grouped-query attention (GQA) is handled directly from the input shapes: query
head ``h`` maps to key head ``h // (Hq // Hkv)``, matching both ``enable_gqa``
and transformers' ``repeat_kv``. Causal masking uses top-left alignment (query
``i`` attends to keys ``0..i``), matching PyTorch's ``is_causal=True``. Returned
tensors are detached from autograd because the kernel's ``lse`` is captured below
autograd, where differentiating through ``exp(S - lse)`` would treat ``lse`` as a
constant. When ``dropout_p > 0``, dropout is applied after these probabilities,
and the provenance note records that ``dropout_p`` value.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
from typing import Any, Callable, Iterator

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from . import kernel
from .kernel import Provenance, Sites, Unavailable

aten = torch.ops.aten

#: Fused SDPA ATen operators mapped to the positional index of their
#: ``compute_log_sumexp`` argument, or ``None`` for operators that always return ``lse``.
_FUSED: dict[OpOverload, int | None] = {
    aten._scaled_dot_product_flash_attention.default: None,
    aten._scaled_dot_product_flash_attention_for_cpu.default: None,
    aten._scaled_dot_product_efficient_attention.default: 4,
    aten._scaled_dot_product_cudnn_attention.default: 4,
    aten._scaled_dot_product_fused_attention_overrideable.default: None,
}

#: Parameter names of ``scaled_dot_product_attention`` in positional order
#: (needed because the C++ builtin has no Python ``inspect.signature``).
_PARAMETERS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")

#: Threshold at or below which a float mask entry is treated as completely masked.
#: Transformers constructs additive masks using the minimum representable value of
#: the model dtype, and ``float16`` has the largest minimum among supported dtypes.
_MASKED = torch.finfo(torch.float16).min


class _Capture(TorchDispatchMode):
    """Dispatch mode that records the fused SDPA operator, its arguments, and its output."""

    def __init__(self) -> None:
        super().__init__()
        self.op: OpOverload | None = None
        self.args: tuple = ()
        self.kwargs: dict = {}
        self.result: Any = None

    def __torch_dispatch__(self, func: OpOverload, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if func in _FUSED:
            self.op, self.args, self.kwargs, self.result = func, args, kwargs, result
        return result


def serve(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Run ``scaled_dot_product_attention`` under `_Capture` and return its output and an `Attention` reader."""
    bound = dict(zip(_PARAMETERS, args))
    bound.update(kwargs)
    with _Capture() as capture:
        output = fn(*args, **kwargs)
    if bound["query"].dim() != 4:
        return output, Unavailable(f"kernel sites need [batch, heads, queries, dim] attention inputs; got {bound['query'].dim()} dims")
    return output, Attention(bound, capture)


@contextmanager
def _exact_fp32() -> Iterator[None]:
    """Run fp32 matrix multiplications at full IEEE precision regardless of the caller's TF32 setting."""
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)


class Attention(Sites):
    """Site reader for a captured `scaled_dot_product_attention` call."""

    names = ("lse", "scores", "probs")

    def __init__(self, bound: dict[str, Any], capture: _Capture) -> None:
        self.q: torch.Tensor = bound["query"].detach()
        self.k: torch.Tensor = bound["key"].detach()
        mask = bound.get("attn_mask")
        self.mask = None if mask is None else mask.detach()
        self.is_causal = bool(bound.get("is_causal", False))
        scale = bound.get("scale")
        self.scale = self.q.shape[-1] ** -0.5 if scale is None else scale
        self.dropout = bound.get("dropout_p", 0.0)
        self.capture = capture
        # Set to True if a second launch with compute_log_sumexp=True produced a different output from the first launch.
        self.disagreed = False

    @property
    def op_name(self) -> str:
        return "math backend (fp32)" if self.capture.op is None else str(self.capture.op).removesuffix(".default")

    @cached_property
    def exported(self) -> torch.Tensor | None:
        """Return the kernel's exported fp32 logsumexp of shape ``[B, H, Lq]``, or ``None`` if unavailable."""
        capture = self.capture
        if capture.op is None:
            return None
        flag = _FUSED[capture.op]
        result = capture.result
        if flag is not None and not capture.args[flag]:
            # This backend only writes logsumexp when requested. Re-run with the
            # flag enabled and use the result only if the attention output matches
            # the original call bit for bit.
            args = (*capture.args[:flag], True, *capture.args[flag + 1 :])
            with torch.no_grad():
                result = capture.op(*args, **capture.kwargs)
            if not torch.equal(result[0], capture.result[0]):
                self.disagreed = True
                return None
        if result[1].numel() == 0:
            # Custom or overrideable backends (such as XPU) may return an empty logsumexp tensor.
            return None
        lse = result[1].float()
        # cuDNN returns shape [B, H, Lq, 1], and the memory-efficient kernel pads
        # the query dimension to a multiple of 32. Trim both to [B, H, Lq].
        if lse.dim() == 4:
            lse = lse.squeeze(-1)
        return lse[..., : self.q.shape[2]]

    def _note(self, formula: str) -> str:
        if self.dropout:
            return f"{formula}; dropout_p={self.dropout} is applied after these probabilities"
        return formula

    def _axes(self, spec: dict[str, Any], site: str, allowed: tuple[str, ...]) -> tuple:
        """Resolve query heads, corresponding KV heads, query row indices, and key column indices from ``spec``."""
        kernel.check_axes(spec, site, allowed)
        device = self.q.device
        heads = kernel.positions(kernel.indices(spec.get("heads"), self.q.shape[1], device), self.q.shape[1], device)
        kv_heads = heads // (self.q.shape[1] // self.k.shape[1])
        rows = kernel.indices(spec.get("query"), self.q.shape[2], device)
        cols = kernel.indices(spec.get("key"), self.k.shape[2], device)
        return heads, kv_heads, rows, cols

    def _mask(self, heads: torch.Tensor, rows: slice | torch.Tensor, cols: slice | torch.Tensor) -> torch.Tensor | None:
        """Return the explicit ``attn_mask`` sliced to ``(heads, rows, cols)`` as an additive fp32 bias, or ``None``."""
        if self.mask is None:
            return None
        mask = self.mask.reshape((1,) * (4 - self.mask.dim()) + tuple(self.mask.shape))
        # Slice before casting to float32 so requesting a small slice does not copy the full mask.
        # Dimensions of size 1 broadcast across the axis and are left unsliced.
        for dim, index in ((1, heads), (2, rows), (3, cols)):
            if mask.shape[dim] > 1:
                mask = kernel.select(mask, dim, index)
        if mask.dtype is torch.bool:
            return torch.zeros(mask.shape, device=mask.device).masked_fill(~mask, float("-inf"))
        return mask.float()

    def _bias(self, heads: torch.Tensor, rows: slice | torch.Tensor, cols: slice | torch.Tensor) -> torch.Tensor | None:
        """Return the combined explicit and causal additive mask for the slice, or ``None`` if unmasked."""
        bias = self._mask(heads, rows, cols)
        if self.is_causal:
            row = kernel.positions(rows, self.q.shape[2], self.q.device)
            col = kernel.positions(cols, self.k.shape[2], self.q.device)
            causal = torch.zeros(len(row), len(col), device=self.q.device).masked_fill(col[None, :] > row[:, None], float("-inf"))
            bias = causal if bias is None else bias + causal
        return bias

    def _scores(self, heads: torch.Tensor, kv_heads: torch.Tensor, rows: slice | torch.Tensor, cols: slice | torch.Tensor) -> torch.Tensor:
        q = kernel.select(kernel.select(self.q, 1, heads), 2, rows).float()
        k = kernel.select(kernel.select(self.k, 1, kv_heads), 2, cols).float()
        with _exact_fp32():
            scores = (q @ k.transpose(-1, -2)) * self.scale
        bias = self._bias(heads, rows, cols)
        return scores if bias is None else scores + bias

    def _lse(self, heads: torch.Tensor, kv_heads: torch.Tensor, rows: slice | torch.Tensor) -> tuple[torch.Tensor, Provenance]:
        if self.exported is not None:
            lse = kernel.select(kernel.select(self.exported, 1, heads), 2, rows)
            return lse, Provenance("exact export", self.op_name, self._note("the kernel's own per-row logsumexp, fp32"))
        lse = torch.logsumexp(self._scores(heads, kv_heads, rows, slice(None)), dim=-1)
        why = "its second launch disagreed with the model's output" if self.disagreed else "it exported none"
        return lse, Provenance("replay", self.op_name, self._note(f"logsumexp of fp32 scores computed beside the kernel; {why}"))

    def _shape(self, heads: torch.Tensor, rows: slice | torch.Tensor, cols: slice | torch.Tensor) -> tuple[int, ...]:
        return (
            self.q.shape[0],
            heads.numel(),
            len(kernel.positions(rows, self.q.shape[2], self.q.device)),
            len(kernel.positions(cols, self.k.shape[2], self.q.device)),
        )

    def lse(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        heads, kv_heads, rows, _ = self._axes(spec, "lse", ("heads", "query"))
        lse, provenance = self._lse(heads, kv_heads, rows)
        return kernel.squeeze(lse, spec, {"heads": 1, "query": 2}), provenance

    def scores(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        heads, kv_heads, rows, cols = self._axes(spec, "scores", ("heads", "query", "key"))
        kernel.warn_if_large(self._shape(heads, rows, cols), "scores", "heads=, query= or key=")
        scores = self._scores(heads, kv_heads, rows, cols)
        provenance = Provenance("exact reconstruction", self.op_name, self._note("q . k * scale in fp32 from the kernel's inputs, mask added"))
        return kernel.squeeze(scores, spec, {"heads": 1, "query": 2, "key": 3}), provenance

    def probs(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        heads, kv_heads, rows, cols = self._axes(spec, "probs", ("heads", "query", "key"))
        kernel.warn_if_large(self._shape(heads, rows, cols), "probs", "heads=, query= or key=")
        lse, lse_provenance = self._lse(heads, kv_heads, rows)
        probs = torch.exp(self._scores(heads, kv_heads, rows, cols) - lse.unsqueeze(-1))
        if self.mask is not None:
            # When every key in a query row is masked (for example, a padding
            # query), different backends write different values into lse. Zero
            # those rows explicitly since no key received attention.
            whole = self._mask(heads, rows, slice(None))
            probs = probs.masked_fill((whole <= _MASKED).all(-1, keepdim=True), 0.0)
        kind = "exact reconstruction" if lse_provenance.kind == "exact export" else "replay"
        provenance = Provenance(kind, self.op_name, self._note("exp(scores - lse) in fp32, the kernel backward's own formula"))
        return kernel.squeeze(probs, spec, {"heads": 1, "query": 2, "key": 3}), provenance


kernel.register(torch.nn.functional.scaled_dot_product_attention, serve)
