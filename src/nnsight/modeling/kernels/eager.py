"""Attention probabilities, scores, and logsumexp from transformers' eager attention functions.

Under ``attn_implementation="eager"``, a transformers attention module calls the
``eager_attention_forward`` function of its own modeling module (named
``simple_``, ``vision_``, or ``text_eager_attention_forward`` in a few modules),
which builds the probability matrix in Python and returns it alongside the
attention output. `serve_eager` serves the three attention sites of
intervention/attention.py from that call: ``probs`` is the weights tensor
returned by the function and multiplied with the
values (an ``"exact export"`` in the model dtype), and ``scores`` and ``lse`` are
an fp32 replay of ``q . k * scale`` plus the mask computed beside the call
(``"replay"``, because the function computes its own scores in the model dtype).
``probs`` is always served; ``scores`` and ``lse`` raise ``ValueError`` when the
function receives an extra parameter beyond the seven shared parameters (such as
Gemma's ``softcap`` or T5's ``position_bias``) or when the returned rows sum to
less than one (such as when GPT-OSS includes attention sinks from the module),
since the replay from ``query``, ``key``, and ``attention_mask`` does not include
those terms.
"""

from __future__ import annotations

import inspect
from types import FunctionType
from typing import Any, Callable

import torch

from ...intervention import sites
from ...intervention.attention import Attention
from ...intervention.sites import Provenance, Sites, Unavailable
from ...intervention.source import arguments

#: Shared positional arity of transformers' eager attention functions:
#: ``(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs)``,
#: where ``query``, ``key``, and ``value`` may have a ``_states`` suffix and
#: ``scaling`` and ``dropout`` may appear in either order. Any additional named
#: parameter (such as ``softcap`` or ``position_bias``) alters the scores inside the call.
_EAGER_ARITY = 7


def is_eager_forward(fn: Any) -> bool:
    """Return whether ``fn`` is a Python function named ``*eager_attention_forward`` with the standard seven-parameter prefix."""
    if not isinstance(fn, FunctionType) or not fn.__name__.endswith("eager_attention_forward"):
        return False
    names = list(inspect.signature(fn).parameters)
    return len(names) >= _EAGER_ARITY and sorted(names[5:_EAGER_ARITY]) == ["dropout", "scaling"]


def serve_eager(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Run a transformers eager attention function and pair its output with an `Eager` site reader."""
    output = fn(*args, **kwargs)
    bound = arguments(fn, args, kwargs)
    names = list(inspect.signature(fn).parameters)
    weights = output[1] if isinstance(output, tuple) and len(output) == 2 else None
    if not isinstance(weights, torch.Tensor) or weights.dim() != 4 or bound[names[1]].dim() != 4:
        return output, Unavailable("eager attention returned no [batch, heads, queries, keys] weights")
    return output, Eager(fn, bound, weights)


class Eager(Attention):
    """Site reader for a transformers ``eager_attention_forward`` call.

    ``probs`` serves the weights tensor returned by the call. ``scores`` and ``lse``
    recompute ``q . k * scale`` plus ``attention_mask`` in fp32 using `Attention`'s
    slicing and masking helpers and carry provenance kind ``"replay"`` because the
    call computes its own scores in the model dtype. Because eager masks are additive
    and already include causal masking, ``is_causal`` is left unset.
    """

    def __init__(self, fn: Callable, bound: dict[str, Any], weights: torch.Tensor) -> None:
        parameters = inspect.signature(fn).parameters
        names = list(parameters)
        module, query, key, mask = (bound[names[i]] for i in (0, 1, 2, 4))
        # The function applies dropout only when `module.training` is True.
        dropout = bound["dropout"] if module.training else 0.0
        super().__init__({"query": query, "key": key, "attn_mask": mask, "scale": bound["scaling"], "dropout_p": dropout}, None)
        self.weights = weights.detach()
        self.name = f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}"
        self.extras = {
            name: bound[name]
            for name in names[_EAGER_ARITY:]
            if parameters[name].kind is not inspect.Parameter.VAR_KEYWORD and bound[name] is not None
        }

    @property
    def op_name(self) -> str:
        return self.name

    @property
    def exported(self) -> None:
        return None

    def _replayable(self, site: str) -> None:
        """Raise ``ValueError`` if the call modified or normalized the scores using terms outside ``q . k * scale + mask``."""
        if self.extras:
            received = ", ".join(f"{name}=<{type(value).__name__}>" if isinstance(value, torch.Tensor) else f"{name}={value!r}" for name, value in self.extras.items())
            raise ValueError(
                f"kernel site {site!r}: {self.name} received {received}, which the replay of its scores from query, key and the mask does not model; probs is served from the weights it returned"
            )
        if not self.dropout:
            # Without dropout, softmax over the keys sums to 1 on every row (including
            # fully masked rows, which come back uniform). Rows that sum to less were
            # normalized over extra columns such as `module.sinks` in GPT-OSS.
            sums = self.weights.float().sum(-1)
            tolerance = 1e-5 if self.weights.dtype is torch.float32 else 2.0 ** -8
            if (sums - 1).abs().max() > tolerance:
                raise ValueError(
                    f"kernel site {site!r}: the weights {self.name} returned sum to {sums.min().item():.4f} on some rows, so the call normalised over more than the keys (attention sinks); the replay of its scores does not model that; probs is served from the weights it returned"
                )

    def lse(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        self._replayable("lse")
        heads, kv_heads, rows, _ = self._axes(spec, "lse", ("heads", "query"))
        lse = torch.logsumexp(self._scores(heads, kv_heads, rows, slice(None)), dim=-1)
        note = self._note(f"logsumexp of fp32 scores computed beside the call, which keeps its scores in {self.q.dtype}")
        return sites.squeeze(lse, spec, {"heads": 1, "query": 2}), Provenance("replay", self.name, note)

    def scores(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        self._replayable("scores")
        heads, kv_heads, rows, cols = self._axes(spec, "scores", ("heads", "query", "key"))
        sites.warn_if_large(self._shape(heads, rows, cols), "scores", "heads=, query= or key=")
        scores = self._scores(heads, kv_heads, rows, cols)
        note = self._note(f"q . k * scale in fp32 from the call's inputs, mask added; the call keeps its scores in {self.q.dtype}")
        return sites.squeeze(scores, spec, {"heads": 1, "query": 2, "key": 3}), Provenance("replay", self.name, note)

    def probs(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        heads, _, rows, cols = self._axes(spec, "probs", ("heads", "query", "key"))
        # Clone so in-place edits by the caller do not mutate the module's returned weights.
        probs = sites.select(sites.select(sites.select(self.weights, 1, heads), 2, rows), 3, cols).clone()
        note = f"the attention weights the call returned and multiplied with the values, {self.weights.dtype}"
        if self.dropout:
            note = f"{note}; dropout_p={self.dropout} was applied to them (the module is in training mode)"
        return sites.squeeze(probs, spec, {"heads": 1, "query": 2, "key": 3}), Provenance("exact export", self.name, note)
