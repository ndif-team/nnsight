"""Attention probabilities, scores, and logsumexp from FlashAttention 2.

Under ``attn_implementation="flash_attention_2"`` transformers calls the
``flash_attn`` package's functions, or the same functions from the
``kernels-community/flash-attn2`` hub kernel when the package is missing. It
calls ``flash_attn_func`` for a batch without padding and
``flash_attn_varlen_func`` otherwise: for a padded batch, which it first packs
into one row per real token, for sequences packed into one row through
``position_ids``, and when the caller passes ``cu_seq_lens_q`` and
``cu_seq_lens_k``. ``cu_seqlens`` marks where each sequence starts. Neither
function writes the probability matrix. Both compute the per-row logsumexp in
fp32 and return it when called with ``return_attn_probs=True``, in the same
launch that fills the output. `serve` asks for it, returns to the model what the model asked for, and
builds a `Flash` reader that reconstructs ``exp(S - lse)`` from the fp32 scores,
as intervention/attention.py does for PyTorch's SDPA.

Sites are ``lse`` ``[sequences, heads, queries]`` and ``scores`` and ``probs``
``[sequences, heads, queries, keys]``, with ``heads=``, ``query=`` and ``key=``.
The trace's batcher decides which of the kernel's sequences belong to an invoke
(`Batch.rows`): its own rows when the kernel ran one sequence per batch row, and
every sequence otherwise, as for generation's beams or sequences packed into
one row. Because a padded batch runs packed, an invoke with one sequence gets
axes that cover its own tokens and hold no padding positions. An invoke whose
sequences have different lengths gets them aligned at their last query and key
and padded on the left to the longest, the layout of a left-padded batch under
SDPA: padding positions hold score ``-inf`` and probability ``0``, and a padding
query row holds logsumexp ``-inf``.

Causal masking follows FlashAttention 2.1 and later: the mask is aligned to the
bottom right, so query row ``r`` of ``Lq`` attends to keys ``0..r + (Lk - Lq)``.
A call is served only when every argument that changes the scores is one the
reconstruction models; a sliding window, logit soft cap, ALiBi slopes, dropout,
or any other non-default argument makes its sites `Unavailable`. So does a
varlen logsumexp in the padded ``[batch, heads, max_seqlen_q]`` layout that
``flash_attn`` returned before 2.6.0.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, NamedTuple

import torch

from ...intervention import sites
from ...intervention.attention import _exact_fp32
from ...intervention.sites import Batch, Provenance, Sites, Unavailable
from ...intervention.source import arguments

#: Arguments the reconstruction uses or that cannot change scores and probabilities.
#: Any other argument must be at its signature default for the call to be served.
_MODELLED = frozenset(
    {"q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "softmax_scale", "causal", "deterministic", "return_attn_probs"}
)


def _unmodelled(fn: Callable, bound: dict[str, Any]) -> str | None:
    """Name the first argument outside `_MODELLED` that differs from its default, or ``None``."""
    for name, parameter in inspect.signature(fn).parameters.items():
        if name in _MODELLED:
            continue
        value, default = bound[name], parameter.default
        if isinstance(value, torch.Tensor) or not (value is default or value == default):
            return name
    return None


def serve(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Run a FlashAttention 2 function with its logsumexp returned; hand back what the caller asked for and a `Flash` reader."""
    name = fn.__name__
    bound = arguments(fn, args, kwargs)
    if "return_attn_probs" not in bound:
        return fn(*args, **kwargs), Unavailable(f"{name} has no return_attn_probs argument to return its logsumexp; its kernel sites are not served")
    reason = _unmodelled(fn, bound)
    if reason is not None:
        return fn(*args, **kwargs), Unavailable(f"{name}: the call sets {reason}, which the reconstruction does not model; its kernel sites are not served on this call")
    asked = bound["return_attn_probs"]
    # With dropout_p == 0 (checked above) the third element is an empty placeholder, so asking costs nothing.
    result = fn(**{**bound, "return_attn_probs": True})
    output, lse = (result if asked else result[0]), result[1]
    if bound["q"].dim() == 3 and lse.dim() != 2:
        return output, Unavailable(
            f"{name} returned its logsumexp as {tuple(lse.shape)}, the padded layout of flash_attn before 2.6.0; "
            "its kernel sites need flash_attn 2.6.0 or later"
        )
    return output, Flash(name, bound, lse)


class Axes(NamedTuple):
    """One request resolved against a `Flash` call: requested positions on the padded axes, and the packed rows behind them."""

    heads: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    #: Packed query and key rows read for each sequence and position, ``[prompts, rows]`` and ``[prompts, cols]``.
    at_q: torch.Tensor
    at_k: torch.Tensor
    #: Whether each of those positions is one of the sequence's own tokens rather than padding.
    real_q: torch.Tensor
    real_k: torch.Tensor
    #: How many more key positions than query positions the padded axes have, for the causal mask.
    offset: int


class Flash(Sites):
    """Site reader for one ``flash_attn_func`` or ``flash_attn_varlen_func`` call.

    Both calls are held in packed form: ``q`` ``[query rows, heads, dim]``, ``k``
    ``[key rows, kv heads, dim]``, the rows at which each sequence starts and
    ends on each side (``spans_q``, ``spans_k``), and the kernel's logsumexp as
    ``[heads, query rows]``.
    """

    names = ("lse", "scores", "probs")
    narrowed = True

    def __init__(self, name: str, bound: dict[str, Any], lse: torch.Tensor) -> None:
        self.name = name
        q, k = bound["q"].detach(), bound["k"].detach()
        if q.dim() == 4:
            # Dense call: [batch, rows, heads, dim] on both sides and lse [batch, heads, rows].
            self.spans_q = list(range(0, q.shape[0] * q.shape[1] + 1, q.shape[1]))
            self.spans_k = list(range(0, k.shape[0] * k.shape[1] + 1, k.shape[1]))
            q, k = q.flatten(0, 1), k.flatten(0, 1)
            lse = lse.transpose(0, 1).flatten(1)
        else:
            self.spans_q = bound["cu_seqlens_q"].tolist()
            # transformers passes one tensor for both sides on a prefill; reading it once saves a device sync.
            self.spans_k = self.spans_q if bound["cu_seqlens_k"] is bound["cu_seqlens_q"] else bound["cu_seqlens_k"].tolist()
        self.q: torch.Tensor = q
        self.k: torch.Tensor = k
        #: ``[heads, query rows]`` fp32, as the kernel returns it.
        self.exported: torch.Tensor = lse.detach()
        self.scale: float = bound["softmax_scale"] if bound["softmax_scale"] is not None else q.shape[-1] ** -0.5
        self.causal = bool(bound["causal"])

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        if site not in self.names:
            raise ValueError(f"this kernel has no site {site!r}; available: {', '.join(self.names)}")
        return getattr(self, site)(spec, batch)

    def _axes(self, spec: dict[str, Any], batch: Batch | None, site: str, allowed: tuple[str, ...]) -> Axes:
        """Resolve the worker's sequences and the requested heads, query rows and key columns into packed row indices.

        The sequences are aligned at their last query and key and padded on the
        left to the longest, so position ``r`` of a sequence with ``n`` of the
        ``queries`` rows is its own row ``r - (queries - n)``, absent when negative.
        """
        sites.check_axes(spec, site, allowed)
        count = len(self.spans_q) - 1
        prompts = list(range(count)) if batch is None else batch.rows(count)
        lengths_q = [self.spans_q[i + 1] - self.spans_q[i] for i in prompts]
        lengths_k = [self.spans_k[i + 1] - self.spans_k[i] for i in prompts]
        queries, keys = max(lengths_q), max(lengths_k)
        device = self.q.device
        heads = sites.positions(sites.indices(spec.get("heads"), self.q.shape[1], device), self.q.shape[1], device)
        rows = sites.positions(sites.indices(spec.get("query"), queries, device), queries, device)
        cols = sites.positions(sites.indices(spec.get("key"), keys, device), keys, device)
        own_q = rows[None, :] - torch.tensor([queries - n for n in lengths_q], device=device)[:, None]  # [prompts, rows]
        own_k = cols[None, :] - torch.tensor([keys - n for n in lengths_k], device=device)[:, None]  # [prompts, cols]
        first_q = torch.tensor([self.spans_q[i] for i in prompts], device=device)
        first_k = torch.tensor([self.spans_k[i] for i in prompts], device=device)
        return Axes(heads, rows, cols, first_q[:, None] + own_q.clamp(min=0), first_k[:, None] + own_k.clamp(min=0), own_q >= 0, own_k >= 0, keys - queries)

    def _lse(self, axes: Axes) -> torch.Tensor:
        lse = self.exported.index_select(0, axes.heads)[:, axes.at_q].transpose(0, 1).float()  # [prompts, heads, rows]
        return lse.masked_fill(~axes.real_q[:, None, :], float("-inf"))

    def _scores(self, axes: Axes) -> torch.Tensor:
        q = self.q[axes.at_q].index_select(2, axes.heads).float()  # [prompts, rows, heads, dim]
        k = self.k[axes.at_k].index_select(2, axes.heads // (self.q.shape[1] // self.k.shape[1])).float()  # [prompts, cols, heads, dim]
        with _exact_fp32():
            scores = torch.einsum("prhd,pchd->phrc", q, k) * self.scale
        hidden = ~(axes.real_q[:, None, :, None] & axes.real_k[:, None, None, :])
        if self.causal:
            # Aligned at the last token, each sequence's bottom-right mask is the same mask on the padded axes.
            hidden = hidden | (axes.cols[None, :] > axes.rows[:, None] + axes.offset)
        return scores.masked_fill(hidden, float("-inf"))

    def lse(self, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        axes = self._axes(spec, batch, "lse", ("heads", "query"))
        provenance = Provenance("exact export", self.name, "the kernel's own per-row logsumexp, fp32")
        return sites.squeeze(self._lse(axes), spec, {"heads": 1, "query": 2}), provenance

    def scores(self, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        axes = self._axes(spec, batch, "scores", ("heads", "query", "key"))
        sites.warn_if_large((axes.at_q.shape[0], axes.heads.numel(), axes.rows.numel(), axes.cols.numel()), "scores", "heads=, query= or key=")
        note = "q . k * scale in fp32 from the kernel's inputs" + (", causal mask added" if self.causal else "")
        provenance = Provenance("exact reconstruction", self.name, note)
        return sites.squeeze(self._scores(axes), spec, {"heads": 1, "query": 2, "key": 3}), provenance

    def probs(self, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        axes = self._axes(spec, batch, "probs", ("heads", "query", "key"))
        sites.warn_if_large((axes.at_q.shape[0], axes.heads.numel(), axes.rows.numel(), axes.cols.numel()), "probs", "heads=, query= or key=")
        # A padding query row has scores and logsumexp -inf, whose difference is NaN; no key received attention there.
        probs = torch.exp(self._scores(axes) - self._lse(axes).unsqueeze(-1)).masked_fill(~axes.real_q[:, None, :, None], 0.0)
        provenance = Provenance("exact reconstruction", self.name, "exp(scores - lse) in fp32, the kernel backward's own formula")
        return sites.squeeze(probs, spec, {"heads": 1, "query": 2, "key": 3}), provenance
