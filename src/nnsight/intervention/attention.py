"""Attention probabilities, scores and logsumexp from the fused SDPA kernel.

``torch.nn.functional.scaled_dot_product_attention`` dispatches to one of several
fused kernels (FlashAttention, the memory-efficient kernel, cuDNN, a CPU flash
kernel) or to an fp32 math fallback. None of them writes the probability matrix;
every fused one computes a per-row logsumexp for its backward pass, and the public
call drops it. The probabilities the kernel used are then ``exp(S - lse)`` with
``S`` the fp32 scores, which is the formula FlashAttention's own backward
recomputes them by. The reconstruction is exact up to fp32 rounding of ``S`` and
``lse`` (rows sum to 1 within a few ulps of the row's largest ``|S|``: 5e-7 on
GPT-2, 8e-4 on a Qwen2.5-0.5B layer whose scores reach 1700), where an eager
re-implementation under bf16 is off by up to 19% at the top key.

The public call runs under a dispatch mode (`_Capture`) that sees the fused op the
dispatcher chose and its arguments as preprocessed (mask conversion, stride
padding, the GQA expansion), and keeps them. The op runs exactly as the model
called it, so the output is the model's own. The two kernels that write the
logsumexp only on request (efficient, cuDNN) are launched a second time with the
flag on, when a worker asks and only then; the second output is compared with the
first, and if the two differ (observed on the efficient kernel under ``no_grad``
with a stride-0 key head and a bias, at a few sequence lengths) the logsumexp is
recomputed in fp32 instead and the provenance says so. When no fused op fires the
math fallback ran, which computes in fp32 throughout, and the logsumexp is
computed here in fp32 and labelled a replay.

Sites, each taking ``heads=``, ``query=`` and (except ``lse``) ``key=`` as an int,
slice, list or tensor of indices, with torch indexing semantics:

* ``lse``: the kernel's normalizer, ``[batch, heads, queries]``, an exact export.
* ``scores``: ``q . k * scale`` in fp32 with the mask added: ``-inf`` under a
  causal or boolean mask, the mask's own value under a float mask,
  ``[batch, heads, queries, keys]``.
* ``probs``: ``exp(scores - lse)``, exactly ``0`` where masked; a row with every
  key masked (a pad query) is all ``0``; other rows sum to 1.

.. code-block:: python

    with model.trace(prompt):
        p = model.model.layers[6].self_attn.kernel.probs(heads=4, query=-1).save()

Grouped-query attention is read off the shapes: query head ``h`` reads key head
``h // (Hq // Hkv)``, the layout both ``enable_gqa`` and transformers' ``repeat_kv``
use. A causal call is top-left aligned (query ``i`` sees keys ``0..i``), as
``is_causal`` is in torch. The served tensors are detached: the kernel's logsumexp
is taken below autograd, so a gradient through them would treat it as a constant.
Dropout, when on, is applied after these probabilities; the provenance says so.
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

#: The fused ops the dispatcher may choose, mapped to the position of their
#: ``compute_log_sumexp`` argument, or ``None`` for the ones that always write it.
_FUSED: dict[OpOverload, int | None] = {
    aten._scaled_dot_product_flash_attention.default: None,
    aten._scaled_dot_product_flash_attention_for_cpu.default: None,
    aten._scaled_dot_product_efficient_attention.default: 4,
    aten._scaled_dot_product_cudnn_attention.default: 4,
    aten._scaled_dot_product_fused_attention_overrideable.default: None,
}

#: ``scaled_dot_product_attention``'s parameters in order, for binding positionals;
#: the function is a builtin, so ``inspect.signature`` has none to give.
_PARAMETERS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")

#: A float mask at or below this is a masked position: transformers builds additive
#: masks at the model dtype's minimum, and float16's is the largest of those.
_MASKED = torch.finfo(torch.float16).min


class _Capture(TorchDispatchMode):
    """Record the fused op the dispatcher chose, its arguments and its result."""

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
    """Run the public SDPA call as the model made it; return its output and a reader."""
    bound = dict(zip(_PARAMETERS, args))
    bound.update(kwargs)
    with _Capture() as capture:
        output = fn(*args, **kwargs)
    if bound["query"].dim() != 4:
        return output, Unavailable(f"kernel sites need [batch, heads, queries, dim] attention inputs; got {bound['query'].dim()} dims")
    return output, Attention(bound, capture)


@contextmanager
def _exact_fp32() -> Iterator[None]:
    """fp32 matmuls at full precision, whatever TF32 setting the user runs under."""
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)


class Attention(Sites):
    """The three sites over one captured call, computed per requested slice."""

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
        # Set when the kernel's second launch did not reproduce the model's output.
        self.disagreed = False

    @property
    def op_name(self) -> str:
        return "math backend (fp32)" if self.capture.op is None else str(self.capture.op).removesuffix(".default")

    @cached_property
    def exported(self) -> torch.Tensor | None:
        """The kernel's logsumexp, fp32, ``[B, H, Lq]``, or ``None`` when it wrote none."""
        capture = self.capture
        if capture.op is None:
            return None
        flag = _FUSED[capture.op]
        result = capture.result
        if flag is not None and not capture.args[flag]:
            # Written only on request: launch once more with the flag on, and
            # trust the logsumexp only if that launch reproduces the model's output.
            args = (*capture.args[:flag], True, *capture.args[flag + 1 :])
            with torch.no_grad():
                result = capture.op(*args, **capture.kwargs)
            if not torch.equal(result[0], capture.result[0]):
                self.disagreed = True
                return None
        if result[1].numel() == 0:
            # The overrideable op (XPU, custom backends) may return an empty logsumexp.
            return None
        lse = result[1].float()
        # cuDNN writes ``[B, H, Lq, 1]``; the efficient kernel pads the last dim to
        # a multiple of 32. Both are the same statistic.
        if lse.dim() == 4:
            lse = lse.squeeze(-1)
        return lse[..., : self.q.shape[2]]

    def _note(self, formula: str) -> str:
        if self.dropout:
            return f"{formula}; dropout_p={self.dropout} is applied after these probabilities"
        return formula

    def _axes(self, spec: dict[str, Any], site: str, allowed: tuple[str, ...]) -> tuple:
        """Query heads, their key heads, query rows and key columns the spec selects."""
        kernel.check_axes(spec, site, allowed)
        device = self.q.device
        heads = kernel.positions(kernel.indices(spec.get("heads"), self.q.shape[1], device), self.q.shape[1], device)
        kv_heads = heads // (self.q.shape[1] // self.k.shape[1])
        rows = kernel.indices(spec.get("query"), self.q.shape[2], device)
        cols = kernel.indices(spec.get("key"), self.k.shape[2], device)
        return heads, kv_heads, rows, cols

    def _mask(self, heads: torch.Tensor, rows: slice | torch.Tensor, cols: slice | torch.Tensor) -> torch.Tensor | None:
        """The explicit mask over the requested slice as a float bias, or ``None``."""
        if self.mask is None:
            return None
        mask = self.mask.reshape((1,) * (4 - self.mask.dim()) + tuple(self.mask.shape))
        # Select before converting, so a one-row request never copies the whole mask.
        # A size-1 dim broadcasts over the slice and is left alone.
        for dim, index in ((1, heads), (2, rows), (3, cols)):
            if mask.shape[dim] > 1:
                mask = kernel.select(mask, dim, index)
        if mask.dtype is torch.bool:
            return torch.zeros(mask.shape, device=mask.device).masked_fill(~mask, float("-inf"))
        return mask.float()

    def _bias(self, heads: torch.Tensor, rows: slice | torch.Tensor, cols: slice | torch.Tensor) -> torch.Tensor | None:
        """The additive mask for the requested slice, or ``None`` when nothing is masked."""
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
            # A row with every key masked (a pad query): the kernels disagree on
            # what they write there, and no key received any attention.
            whole = self._mask(heads, rows, slice(None))
            probs = probs.masked_fill((whole <= _MASKED).all(-1, keepdim=True), 0.0)
        kind = "exact reconstruction" if lse_provenance.kind == "exact export" else "replay"
        provenance = Provenance(kind, self.op_name, self._note("exp(scores - lse) in fp32, the kernel backward's own formula"))
        return kernel.squeeze(probs, spec, {"heads": 1, "query": 2, "key": 3}), provenance


kernel.register(torch.nn.functional.scaled_dot_product_attention, serve)
