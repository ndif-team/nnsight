"""Attention probabilities, scores, and logsumexp from the FlashAttention 2 package.

Under ``attn_implementation="flash_attention_2"`` transformers calls
``flash_attn_func`` for a batch without padding and ``flash_attn_varlen_func``
for a padded batch, which it first packs into one row per real token with
``cu_seqlens`` marking each prompt's span. Neither function writes the
probability matrix. Both compute the per-row logsumexp in fp32 and return it
when called with ``return_attn_probs=True``, in the same launch that fills the
output. `serve` asks for it, returns to the model what the model asked for, and
builds a `Flash` reader that reconstructs ``exp(S - lse)`` from the fp32 scores,
as intervention/attention.py does for PyTorch's SDPA.

Sites are ``lse`` ``[batch, heads, queries]`` and ``scores`` and ``probs``
``[batch, heads, queries, keys]``, with ``heads=``, ``query=`` and ``key=``.
Because a padded batch runs packed, the query and key axes cover a prompt's own
tokens and hold no padding positions. An invoke whose prompts have different
lengths therefore has no rectangular result and raises ``ValueError``; one
prompt per invoke always works.

Causal masking follows FlashAttention 2.1 and later: the mask is aligned to the
bottom right, so query row ``r`` of ``Lq`` attends to keys ``0..r + (Lk - Lq)``.
A call is served only when every argument that changes the scores is one the
reconstruction models; a sliding window, logit soft cap, ALiBi slopes, dropout,
or any other non-default argument makes its sites `Unavailable`.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

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
    return (result if asked else result[0]), Flash(name, bound, result[1])


class Flash(Sites):
    """Site reader for one ``flash_attn_func`` or ``flash_attn_varlen_func`` call.

    Both calls are held in packed form: ``q`` ``[query rows, heads, dim]``, ``k``
    ``[key rows, kv heads, dim]``, the row at which each prompt starts on each
    side, and the kernel's logsumexp as ``[heads, query rows]``.
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
            self.spans_k = bound["cu_seqlens_k"].tolist()
        self.q: torch.Tensor = q
        self.k: torch.Tensor = k
        #: ``[heads, query rows]`` fp32, as the kernel returns it.
        self.exported: torch.Tensor = lse.detach()
        self.scale: float = bound["softmax_scale"] if bound["softmax_scale"] is not None else q.shape[-1] ** -0.5
        self.causal = bool(bound["causal"])
        self.batch: Batch | None = None

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        self.batch = batch
        return super().read(site, spec, batch)

    def _axes(self, spec: dict[str, Any], site: str, allowed: tuple[str, ...]) -> tuple:
        """Resolve the worker's prompts and the requested heads, query rows and key columns into packed row indices."""
        sites.check_axes(spec, site, allowed)
        start, size = (0, len(self.spans_q) - 1) if self.batch is None else (self.batch.start, self.batch.size)
        prompts = range(start, start + size)
        queries = {self.spans_q[i + 1] - self.spans_q[i] for i in prompts}
        keys = {self.spans_k[i + 1] - self.spans_k[i] for i in prompts}
        if len(queries) != 1 or len(keys) != 1:
            raise ValueError(
                f"kernel sites: {self.name} ran this invoke's {size} prompts packed and their key counts differ ({sorted(keys)}), "
                "so they have no shared [queries, keys] shape; request sites from an invoke per prompt"
            )
        queries, keys = queries.pop(), keys.pop()
        device = self.q.device
        heads = sites.positions(sites.indices(spec.get("heads"), self.q.shape[1], device), self.q.shape[1], device)
        rows = sites.positions(sites.indices(spec.get("query"), queries, device), queries, device)
        cols = sites.positions(sites.indices(spec.get("key"), keys, device), keys, device)
        first_q = torch.tensor([self.spans_q[i] for i in prompts], device=device)
        first_k = torch.tensor([self.spans_k[i] for i in prompts], device=device)
        return heads, rows, cols, first_q[:, None] + rows, first_k[:, None] + cols, keys - queries

    def _lse(self, heads: torch.Tensor, at_q: torch.Tensor) -> torch.Tensor:
        return self.exported.index_select(0, heads)[:, at_q].transpose(0, 1).float()  # [prompts, heads, rows]

    def _scores(self, heads: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, at_q: torch.Tensor, at_k: torch.Tensor, offset: int) -> torch.Tensor:
        q = self.q[at_q].index_select(2, heads).float()  # [prompts, rows, heads, dim]
        k = self.k[at_k].index_select(2, heads // (self.q.shape[1] // self.k.shape[1])).float()  # [prompts, cols, heads, dim]
        with _exact_fp32():
            scores = torch.einsum("prhd,pchd->phrc", q, k) * self.scale
        if self.causal:
            scores = scores.masked_fill(cols[None, :] > rows[:, None] + offset, float("-inf"))
        return scores

    def lse(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        heads, _, _, at_q, _, _ = self._axes(spec, "lse", ("heads", "query"))
        provenance = Provenance("exact export", self.name, "the kernel's own per-row logsumexp, fp32")
        return sites.squeeze(self._lse(heads, at_q), spec, {"heads": 1, "query": 2}), provenance

    def scores(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        heads, rows, cols, at_q, at_k, offset = self._axes(spec, "scores", ("heads", "query", "key"))
        sites.warn_if_large((at_q.shape[0], heads.numel(), rows.numel(), cols.numel()), "scores", "heads=, query= or key=")
        provenance = Provenance("exact reconstruction", self.name, "q . k * scale in fp32 from the kernel's inputs, causal mask added")
        return sites.squeeze(self._scores(heads, rows, cols, at_q, at_k, offset), spec, {"heads": 1, "query": 2, "key": 3}), provenance

    def probs(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        heads, rows, cols, at_q, at_k, offset = self._axes(spec, "probs", ("heads", "query", "key"))
        sites.warn_if_large((at_q.shape[0], heads.numel(), rows.numel(), cols.numel()), "probs", "heads=, query= or key=")
        probs = torch.exp(self._scores(heads, rows, cols, at_q, at_k, offset) - self._lse(heads, at_q).unsqueeze(-1))
        provenance = Provenance("exact reconstruction", self.name, "exp(scores - lse) in fp32, the kernel backward's own formula")
        return sites.squeeze(probs, spec, {"heads": 1, "query": 2, "key": 3}), provenance
