"""Recurrent states from chunked linear-attention and state-space kernels.

A gated delta rule (Qwen3-Next, Qwen3.5), a Kimi delta attention (Kimi Linear,
GLM-5-Next) or a Mamba2 scan runs in chunks: inside a chunk the kernel works with
matrices over the chunk's tokens, and the recurrent state is formed only at chunk
boundaries, as the one value the next chunk needs.
The state at a token inside a chunk is never computed, by any implementation, so
there is nothing to export for it. What the kernel does hold is the state at every
chunk boundary, which is the sufficient statistic: the state at token ``t`` is the
boundary state before its chunk advanced to ``t`` by the kernel's own intra-chunk
formula, the one it applies over the whole chunk to reach the next boundary,
truncated at ``t``.

Two sites on the mixer module, both served in fp32:

* ``chunk_states(chunks=, heads=)``: the state entering each chunk, index ``0`` the
  initial state, ``[batch, chunks, heads, ...]``, plus the final state as one more
  entry when the kernel produced it. An *exact export*: the kernel's own buffer,
  upcast when the kernel kept it in a narrower dtype (``fla``'s chunk buffer is in
  the input dtype; its final state and transformers' buffers are fp32).
* ``states(tokens=, heads=)``: the state after each requested token,
  ``[batch, tokens, heads, ...]``. Exact where the token ends a chunk whose next
  boundary was exported; otherwise *derived*: the preceding boundary advanced in
  fp32 by the kernel's intra-chunk formula over its recorded intermediates (the
  pairwise decays, keys and delta-corrected values of the gated delta rule, the
  same with a per-channel decay for Kimi delta attention; the segment decays,
  discretized inputs and ``B`` of Mamba2), one batched contraction per touched
  chunk. Mamba1's selective scan (Mamba, FalconMamba, Jamba, Zamba) binds the state
  after every token, so every token is an exact export there. The provenance note carries the check of that formula: taken
  to the chunk's end it reproduces the kernel's next boundary, and the note says
  how far off it lands (fp32 rounding).

transformers' reference implementations (``torch_chunk_gated_delta_rule``,
``chunk_kimi_delta_attention``, ``mamba2_chunk_scan``, ``mamba_selective_scan``, and
the per-token ``torch_recurrent_gated_delta_rule``, ``recurrent_kimi_delta_attention``
and the two ``selective_state_update`` functions used in decoding) are Python. The chunked ones
bind the boundary states and their intra-chunk intermediates to locals as they go;
their serve functions run them instrumented (the same instrumentation ``.source``
uses), record those bindings, and return the output of the instrumented run, which
is the same arithmetic in the same order. The two ``selective_state_update``
functions write the new state into the cache tensor they were handed, so their
serve function copies that tensor before and after.
``fla``'s chunked kernel computes the chunk states in its own Python forward
(``chunk_gated_delta_rule_fwd_h``) and drops them after the output is formed; a
recording stand-in for that function, installed once per process, hands them to
the serve function that is active on the calling thread; its intra-chunk
intermediates are not recorded, so a state inside one of its chunks is the boundary
stepped token by token by the per-token recurrence. Its fused per-token kernel
exports the final state only, which in decoding is the one state there is.
The entries are registered by `TransformersModel` for the kernels its model's
modeling file reaches (see `REFERENCE` and `LIBRARY`); the registry is
process-wide.

.. code-block:: python

    mixer = model.model.layers[0].linear_attn
    with model.trace(prompt):
        boundaries = mixer.kernel.chunk_states().save()        # [batch, chunks + 1, heads, K, V]
        s_100 = mixer.kernel.states(tokens=100).save()         # [batch, heads, K, V]

    s_100.provenance   # Provenance(kind='derived', kernel='torch_chunk_gated_delta_rule', note='... lands within 0.00e+00 relative of ...')
"""

from __future__ import annotations

import inspect
import sys
import threading
from types import ModuleType
from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import kernel
from .kernel import Provenance, Sites, Unavailable
from .source import arguments, record

#: The epsilon transformers' and fla's ``l2norm`` add under the square root.
_L2NORM_EPS = 1e-6

#: ``advance(chunk, boundary, offsets)``: the fp32 ``[batch, heads, ...]`` state entering
#: ``chunk`` taken to each of ``offsets`` (sorted positions inside the chunk), as
#: ``[batch, len(offsets), heads, ...]``; ``None`` when the kernel exports the state
#: after every token and nothing inside a chunk is ever asked for.
Advance = Callable[[int, torch.Tensor, list[int]], torch.Tensor] | None

#: ``step(state, t)`` advances a ``[batch, heads, ...]`` fp32 state across token ``t``.
Step = Callable[[torch.Tensor, int], torch.Tensor]


def _states(fn: Callable, args: tuple, kwargs: dict, name: str) -> tuple[Any, list[torch.Tensor]]:
    """Run ``fn`` instrumented; its output and the distinct tensors bound to ``name``."""
    output, bound = record(fn, args, kwargs, (name,))
    return output, bound[name]


class Recurrence(Sites):
    """Chunk-boundary states and the per-token states derived from them.

    ``boundaries`` is ``[batch, n, heads, ...]``: the state entering each of the
    first ``n`` chunks of ``chunk_size`` tokens over ``length`` tokens, in the dtype
    the kernel kept; ``final`` the state after the last chunk when the kernel
    produced it, or ``None``. Build one through `recurrence`, which checks the count.
    """

    names = ("chunk_states", "states")

    def __init__(
        self,
        boundaries: torch.Tensor,
        final: torch.Tensor | None,
        chunk_size: int,
        length: int,
        advance: Advance,
        op_name: str,
    ) -> None:
        self.boundaries = boundaries.detach()
        self.final = None if final is None else final.detach()
        self.chunk_size = chunk_size
        self.length = length
        self.advance = advance
        self.op_name = op_name
        self.chunks = -(-length // chunk_size)
        self.count = boundaries.shape[1] + (final is not None)

    def _boundary(self, index: int) -> torch.Tensor:
        """The state entering chunk ``index`` (``self.chunks`` for the final state), fp32."""
        if index == self.boundaries.shape[1]:
            return self.final.float()
        return self.boundaries[:, index].float()

    def _stack(self, states: list[torch.Tensor], spec: dict[str, Any], axis: str) -> torch.Tensor:
        """``[batch, len(states), heads, ...]`` narrowed to the requested heads, a fresh tensor."""
        shape = self.boundaries.shape
        if states:
            value = torch.stack(states, 1)
        else:
            value = self.boundaries.new_zeros((shape[0], 0, *shape[2:]), dtype=torch.float32)
        value = kernel.select(value, 2, kernel.indices(spec.get("heads"), shape[2], value.device))
        return kernel.squeeze(value, spec, {"heads": 2, axis: 1}).clone()

    def chunk_states(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        kernel.check_axes(spec, "chunk_states", ("chunks", "heads"))
        positions = kernel.positions(kernel.indices(spec.get("chunks"), self.count), self.count).tolist()
        value = self._stack([self._boundary(i) for i in positions], spec, "chunks")
        kept = f"kept in {self.boundaries.dtype}" if self.boundaries.dtype is not torch.float32 else "fp32"
        final = f", index {self.chunks} final" if self.count > self.chunks else "; the kernel returned no final state"
        if self.chunk_size == 1:
            note = f"the kernel's own state after each token, {kept}, index 0 initial{final}"
        else:
            note = f"the state entering each chunk of {self.chunk_size} tokens, {kept}, index 0 initial{final}"
        return value, Provenance("exact export", self.op_name, note)

    def states(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        kernel.check_axes(spec, "states", ("tokens", "heads"))
        positions = kernel.positions(kernel.indices(spec.get("tokens"), self.length), self.length).tolist()
        kernel.warn_if_large((self.boundaries.shape[0], len(positions), *self.boundaries.shape[2:]), "states", "tokens=, heads=")
        by_chunk: dict[int, set[int]] = {}
        for token in positions:
            by_chunk.setdefault(token // self.chunk_size, set()).add(token)
        computed: dict[int, torch.Tensor] = {}
        drifts: list[float] = []
        derived = False
        for chunk, tokens in sorted(by_chunk.items()):
            start = chunk * self.chunk_size
            end = min(start + self.chunk_size, self.length)
            exported_next = chunk + 1 < self.count
            if exported_next and end - 1 in tokens:
                computed[end - 1] = self._boundary(chunk + 1)
                tokens = tokens - {end - 1}
            if not tokens:
                continue
            derived = True
            if self.advance is None:
                raise ValueError(
                    f"{self.op_name} exports the state after its token only; "
                    f"tokens={sorted(tokens)} inside it cannot be derived"
                )
            # One contraction per touched chunk. The chunk's last position rides
            # along when the kernel exported the boundary after it, as the check.
            offsets = sorted(t - start for t in tokens)
            last = end - 1 - start
            if exported_next and last not in offsets:
                offsets.append(last)
            states = self.advance(chunk, self._boundary(chunk), offsets)
            for i, offset in enumerate(offsets):
                if start + offset in tokens:
                    computed[start + offset] = states[:, i]
            if exported_next:
                reference = self._boundary(chunk + 1)
                scale = reference.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
                drifts.append(((states[:, -1] - reference).abs().max() / scale).item())
        value = self._stack([computed[token] for token in positions], spec, "tokens")
        if not derived:
            note = "the kernel's own state after each token" if self.chunk_size == 1 else "every requested token ends a chunk; the kernel's own boundary states"
            return value, Provenance("exact export", self.op_name, note)
        note = "the preceding chunk-boundary state advanced to the token in fp32 by the kernel's intra-chunk formula, over the whole batch"
        if drifts:
            note += f"; taken to each touched chunk's end it lands within {max(drifts):.2e} relative of the kernel's next boundary"
        else:
            note += "; the kernel exported no boundary after the touched chunk to check it against"
        return value, Provenance("derived", self.op_name, note)


def recurrence(
    boundaries: torch.Tensor, final: torch.Tensor | None, chunk_size: int, length: int, advance: Advance, op_name: str
) -> Sites:
    """A `Recurrence` over what a kernel exported, or `Unavailable` when the count does not fit the call."""
    chunks = -(-length // chunk_size)
    count = boundaries.shape[1] + (final is not None)
    if count not in (chunks, chunks + 1):
        return Unavailable(
            f"{op_name} produced {count} chunk states for {chunks} chunks of {chunk_size} tokens; "
            f"its forward no longer has the shape this reader expects"
        )
    return Recurrence(boundaries, final, chunk_size, length, advance, op_name)


# ---------------------------------------------------------------------------
# transformers' reference kernels: the gated delta rule (Qwen3-Next, Qwen3.5),
# Kimi delta attention (Kimi Linear, GLM-5-Next), Mamba2's scan, Mamba1's scan
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + _L2NORM_EPS)


def _chunked(x: torch.Tensor, dim: int, chunk_size: int) -> torch.Tensor:
    """``x`` zero-padded along ``dim`` to a multiple of ``chunk_size`` and split there into ``[..., chunks, chunk_size, ...]``."""
    length = x.shape[dim]
    pad = (chunk_size - length % chunk_size) % chunk_size
    if pad:
        widths = [0, 0] * (x.dim() - 1 - dim) + [0, pad]
        x = F.pad(x, widths)
    return x.reshape(*x.shape[:dim], -1, chunk_size, *x.shape[dim + 1 :])


def _walk(step: Step, chunk_size: int) -> Advance:
    """An `Advance` that steps the boundary token by token with ``step``, for a kernel whose intra-chunk intermediates are not recorded."""

    def advance(chunk: int, boundary: torch.Tensor, offsets: list[int]) -> torch.Tensor:
        state, states, start = boundary, [], chunk * chunk_size
        for offset in range(offsets[-1] + 1):
            state = step(state, start + offset)
            if offset in offsets:
                states.append(state)
        return torch.stack(states, 1)

    return advance


def _gated_delta_advance(
    key: torch.Tensor, pairwise_decay: torch.Tensor, new_values: torch.Tensor, k_cumdecay: torch.Tensor, cum_decay: torch.Tensor
) -> Advance:
    """The chunked gated delta rule's own intra-chunk formula, truncated at a position.

    Over a chunk the kernel forms ``v_new = new_values - k_cumdecay @ S_0`` (the
    delta-corrected values given the boundary state) and reaches the next boundary
    as ``exp(G_last) S_0 + sum_j exp(G_last - G_j) k_j^T v_new_j`` with ``G`` the
    cumulative decay. The state after position ``p`` is the same sum stopped at
    ``p``: ``exp(G_p) S_0 + sum_{j <= p} exp(G_p - G_j) k_j^T v_new_j``, and
    ``exp(G_p - G_j)`` masked to ``j <= p`` is the kernel's ``pairwise_decay``.
    ``key`` is the kernel's key as it used it (fp32, l2-normalized when it does
    that), ``[batch, heads, chunks, chunk, K]``.
    """

    def advance(chunk: int, boundary: torch.Tensor, offsets: list[int]) -> torch.Tensor:
        index = torch.tensor(offsets, device=boundary.device)
        v_new = new_values[:, :, chunk] - k_cumdecay[:, :, chunk] @ boundary
        decay = cum_decay[:, :, chunk, index].exp()
        states = decay[..., None, None] * boundary[:, :, None] + torch.einsum(
            "bhnj,bhjk,bhjv->bhnkv", pairwise_decay[:, :, chunk, index], key[:, :, chunk], v_new
        )
        return states.transpose(1, 2)

    return advance


def _mamba2_advance(hidden_states: torch.Tensor, B: torch.Tensor, A_cumsum: torch.Tensor, L: torch.Tensor) -> Advance:
    """Mamba2's chunk scan formula, truncated at a position.

    The kernel reaches the next boundary as ``exp(A_p) S_0 + sum_j exp(A_last - A_j)
    (x_j dt_j) B_j^T`` with ``A`` the cumulative discretized decay; the state after
    position ``p`` is that sum stopped at ``p`` with ``exp(A_p - A_j)``, which masked
    to ``j <= p`` is the kernel's ``L``. ``hidden_states`` is ``x * dt`` and ``B`` is
    repeated to the heads, both ``[batch, chunks, chunk, heads, ...]``.
    """

    def advance(chunk: int, boundary: torch.Tensor, offsets: list[int]) -> torch.Tensor:
        index = torch.tensor(offsets, device=boundary.device)
        decay = A_cumsum[:, :, chunk, index].exp()
        states = decay[..., None, None] * boundary[:, :, None] + torch.einsum(
            "bhts,bshp,bshn->bhtpn", L[:, :, chunk, index], hidden_states[:, chunk], B[:, chunk]
        )
        return states.transpose(1, 2)

    return advance


def _kda_advance(
    key: torch.Tensor, decay_mask: torch.Tensor, values: torch.Tensor, k_cumdecay: torch.Tensor, cum_decay: torch.Tensor
) -> Advance:
    """Kimi delta attention's intra-chunk formula, truncated at a position.

    The gated delta rule with the decay a vector over the key channels: the kernel
    reaches the next boundary as ``exp(G_last) * S_0 + sum_j (k_j * exp(G_last - G_j))^T
    v_new_j`` with ``G`` the cumulative decay, ``[chunk, K]``, broadcast down the
    rows of the state. The state after position ``p`` is that sum stopped at ``p``,
    and ``exp(G_p - G_j)`` masked to ``j <= p`` is the kernel's ``decay_mask``,
    ``[batch, heads, chunks, chunk, chunk, K]``. ``key`` is the kernel's key as it
    used it, ``[batch, heads, chunks, chunk, K]``; ``values`` its ``value`` after the
    WY transform, ``[batch, heads, chunks, chunk, V]``.
    """

    def advance(chunk: int, boundary: torch.Tensor, offsets: list[int]) -> torch.Tensor:
        index = torch.tensor(offsets, device=boundary.device)
        v_new = values[:, :, chunk] - k_cumdecay[:, :, chunk] @ boundary
        decay = cum_decay[:, :, chunk, index].exp()
        states = decay[..., None] * boundary[:, :, None] + torch.einsum(
            "bhnjk,bhjk,bhjv->bhnkv", decay_mask[:, :, chunk, index], key[:, :, chunk], v_new
        )
        return states.transpose(1, 2)

    return advance


def _gated_delta_step(key: torch.Tensor, value: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor, l2norm: bool) -> Step:
    """The gated delta rule's per-token recurrence in fp32, over ``[batch, tokens, heads, ...]`` inputs.

    ``S_t = S' + k_t (beta_t (v_t - k_t^T S'))^T`` with ``S' = S_{t-1} exp(g_t)``, the
    update ``torch_recurrent_gated_delta_rule`` performs; the query enters the
    output only, so it is not needed here. ``beta`` is the gate as the kernel used
    it (after its sigmoid, when the kernel applies one). A key with fewer heads
    than the value (fla's grouped value heads) is repeated to the value's heads,
    the layout the kernel uses.
    """
    key = key.detach().transpose(1, 2).float()
    value = value.detach().transpose(1, 2).float()
    decay = decay.detach().transpose(1, 2).float()
    beta = beta.detach().transpose(1, 2).float()
    if key.shape[1] != value.shape[1]:
        key = key.repeat_interleave(value.shape[1] // key.shape[1], dim=1)
    if l2norm:
        key = _l2norm(key)

    def step(state: torch.Tensor, t: int) -> torch.Tensor:
        k_t, v_t = key[:, :, t], value[:, :, t]
        state = state * decay[:, :, t].exp()[..., None, None]
        memory = (state * k_t.unsqueeze(-1)).sum(-2)
        delta = (v_t - memory) * beta[:, :, t].unsqueeze(-1)
        return state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

    return step


def _shapes(fn: Callable, expected: dict[str, tuple[torch.Tensor | None, tuple[int, ...]]]) -> Unavailable | None:
    """`Unavailable` naming the first recorded intermediate whose shape is not the expected one, else ``None``."""
    for name, (tensor, shape) in expected.items():
        if tensor is None or tuple(tensor.shape) != shape:
            found = "nothing" if tensor is None else f"shape {tuple(tensor.shape)}"
            return Unavailable(f"{fn.__name__} bound {name} as {found}, expected {shape}; its intra-chunk formula no longer has the shape this reader expects")
    return None


def _last(recorded: dict[str, list[torch.Tensor]], name: str) -> torch.Tensor | None:
    return recorded[name][-1] if recorded[name] else None


def serve_gated_delta_chunk(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """transformers' ``torch_chunk_gated_delta_rule``: boundary states from its chunk loop, and the
    intra-chunk intermediates it formed on the way (`_gated_delta_advance`)."""
    output, recorded = record(fn, args, kwargs, ("last_recurrent_state", "key", "pairwise_decay", "new_values", "k_cumdecay", "cum_decay"))
    states = recorded["last_recurrent_state"]
    bound = arguments(fn, args, kwargs)
    length, chunk_size = bound["key"].shape[1], bound["chunk_size"]
    chunks = -(-length // chunk_size)
    if len(states) != chunks + 1:
        return output, Unavailable(
            f"{fn.__name__} bound the recurrent state {len(states)} times over {chunks} chunks; "
            f"its loop no longer has the shape this reader expects"
        )
    batch, heads, head_dim = states[0].shape[0], states[0].shape[1], states[0].shape[2]
    # The first binding of ``key`` is the key as the kernel uses it, [batch, heads, length, K],
    # fp32 and normalized when the kernel normalizes; the kernel pads and chunks it later.
    key = recorded["key"][0] if recorded["key"] else None
    expected = {
        "key": (key, (batch, heads, length, head_dim)),
        "pairwise_decay": (_last(recorded, "pairwise_decay"), (batch, heads, chunks, chunk_size, chunk_size)),
        "new_values": (_last(recorded, "new_values"), (batch, heads, chunks, chunk_size, states[0].shape[3])),
        "k_cumdecay": (_last(recorded, "k_cumdecay"), (batch, heads, chunks, chunk_size, head_dim)),
        "cum_decay": (_last(recorded, "cum_decay"), (batch, heads, chunks, chunk_size)),
    }
    if (unavailable := _shapes(fn, expected)) is not None:
        return output, unavailable
    advance = _gated_delta_advance(
        _chunked(expected["key"][0].detach(), 2, chunk_size),
        *(expected[name][0].detach() for name in ("pairwise_decay", "new_values", "k_cumdecay", "cum_decay")),
    )
    return output, Recurrence(torch.stack(states, 1), None, chunk_size, length, advance, fn.__name__)


def serve_gated_delta_recurrent(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """transformers' ``torch_recurrent_gated_delta_rule`` and ``recurrent_kimi_delta_attention``
    (decoding): every token is a chunk.

    Both loops bind the state twice per token, decayed then updated; the updated
    one is the state after the token.
    """
    output, states = _states(fn, args, kwargs, "last_recurrent_state")
    bound = arguments(fn, args, kwargs)
    length = bound["key"].shape[1]
    if len(states) != 1 + 2 * length:
        return output, Unavailable(
            f"{fn.__name__} bound the recurrent state {len(states)} times over {length} tokens; "
            f"its loop no longer has the shape this reader expects"
        )
    return output, Recurrence(torch.stack([states[0], *states[2::2]], 1), None, 1, length, None, fn.__name__)


def serve_kda_chunk(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """transformers' ``chunk_kimi_delta_attention``: boundary states from its chunk loop, and the
    intra-chunk intermediates it formed on the way (`_kda_advance`)."""
    output, recorded = record(fn, args, kwargs, ("last_recurrent_state", "key", "value", "g", "decay_mask", "k_cumdecay"))
    states = recorded["last_recurrent_state"]
    bound = arguments(fn, args, kwargs)
    length, chunk_size = bound["key"].shape[1], bound["chunk_size"]
    chunks = -(-length // chunk_size)
    if len(states) != chunks + 1:
        return output, Unavailable(
            f"{fn.__name__} bound the recurrent state {len(states)} times over {chunks} chunks; "
            f"its loop no longer has the shape this reader expects"
        )
    batch, heads, head_dim, value_dim = states[0].shape
    padded = chunks * chunk_size
    # The last bindings: ``key`` padded, fp32 and normalized when the kernel normalizes,
    # [batch, heads, padded, K]; ``value`` after the WY transform and ``g`` after the
    # cumulative sum, both chunked.
    expected = {
        "key": (_last(recorded, "key"), (batch, heads, padded, head_dim)),
        "value": (_last(recorded, "value"), (batch, heads, chunks, chunk_size, value_dim)),
        "g": (_last(recorded, "g"), (batch, heads, chunks, chunk_size, head_dim)),
        "decay_mask": (_last(recorded, "decay_mask"), (batch, heads, chunks, chunk_size, chunk_size, head_dim)),
        "k_cumdecay": (_last(recorded, "k_cumdecay"), (batch, heads, chunks, chunk_size, head_dim)),
    }
    if (unavailable := _shapes(fn, expected)) is not None:
        return output, unavailable
    advance = _kda_advance(
        _chunked(expected["key"][0].detach(), 2, chunk_size),
        *(expected[name][0].detach() for name in ("decay_mask", "value", "k_cumdecay", "g")),
    )
    return output, Recurrence(torch.stack(states, 1), None, chunk_size, length, advance, fn.__name__)


def serve_mamba1_scan(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """transformers' ``mamba_selective_scan`` (Mamba, FalconMamba, Jamba, Zamba): its loop binds the
    state after every token, ``[batch, channels, state_size]``, so every token is a chunk of one
    and an exact export. The ``heads`` axis of the sites is the channel axis.

    Its ``mambapy`` and associative-scan paths bind the state once, as the final one.
    """
    output, states = _states(fn, args, kwargs, "ssm_state")
    bound = arguments(fn, args, kwargs)
    length = bound["hidden_states"].shape[2]
    if len(states) != length + 1:
        return output, Unavailable(
            f"{fn.__name__} bound the recurrent state {len(states)} times over {length} tokens; "
            f"it took a path that keeps the final state only, or its loop no longer has the shape this reader expects"
        )
    return output, Recurrence(torch.stack([state.float() for state in states], 1), None, 1, length, None, fn.__name__)


def serve_mamba2_chunk(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """transformers' ``mamba2_chunk_scan``: the states entering each chunk, plus the final one, and the
    intra-chunk intermediates it formed on the way (`_mamba2_advance`)."""
    output, recorded = record(fn, args, kwargs, ("new_states", "hidden_states", "B", "A_cumsum", "L"))
    states = recorded["new_states"]
    bound = arguments(fn, args, kwargs)
    length, chunk_size = bound["hidden_states"].shape[1], bound["chunk_size"]
    if len(states) != 1:
        return output, Unavailable(f"{fn.__name__} bound new_states {len(states)} times; expected once")
    batch, heads, head_dim, state_dim = states[0].shape[0], states[0].shape[2], states[0].shape[3], states[0].shape[4]
    chunks = -(-length // chunk_size)
    # The last bindings before the kernel chunks them: ``hidden_states`` is ``x * dt`` and
    # ``B`` is repeated to the heads, both [batch, length, heads, ...].
    expected = {
        "hidden_states": (_last(recorded, "hidden_states"), (batch, length, heads, head_dim)),
        "B": (_last(recorded, "B"), (batch, length, heads, state_dim)),
        "A_cumsum": (_last(recorded, "A_cumsum"), (batch, heads, chunks, chunk_size)),
        "L": (_last(recorded, "L"), (batch, heads, chunks, chunk_size, chunk_size)),
    }
    if (unavailable := _shapes(fn, expected)) is not None:
        return output, unavailable
    advance = _mamba2_advance(
        _chunked(expected["hidden_states"][0].detach(), 1, chunk_size),
        _chunked(expected["B"][0].detach(), 1, chunk_size),
        expected["A_cumsum"][0].detach(),
        expected["L"][0].detach(),
    )
    return output, recurrence(states[0], None, chunk_size, length, advance, fn.__name__)


def serve_mamba2_decode(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """transformers' ``mamba2_selective_state_update`` and ``mamba_selective_state_update`` (decoding):
    the cache state before and after the token.

    Both write the new state into the ``state`` tensor they were handed, so both
    states are copied out around the call.
    """
    bound = arguments(fn, args, kwargs)
    before = bound["state"].detach().clone()
    output = fn(*args, **kwargs)
    after = bound["state"].detach().clone()
    return output, Recurrence(torch.stack([before, after], 1), None, 1, 1, None, fn.__name__)


# ---------------------------------------------------------------------------
# fla's kernels
# ---------------------------------------------------------------------------

_fla_capture = threading.local()
_fla_installed: set[str] = set()
_fla_lock = threading.Lock()


def _install_fla_recorder(module: ModuleType) -> None:
    """Replace ``module.chunk_gated_delta_rule_fwd_h`` once with a stand-in that also hands
    its result to the serve function active on the calling thread, if any."""
    with _fla_lock:
        if module.__name__ in _fla_installed:
            return
        inner = module.chunk_gated_delta_rule_fwd_h

        def recording(*inner_args: Any, **inner_kwargs: Any) -> Any:
            result = inner(*inner_args, **inner_kwargs)
            captured = vars(_fla_capture).get("results")
            if captured is not None:
                captured.append(result)
            return result

        module.chunk_gated_delta_rule_fwd_h = recording
        _fla_installed.add(module.__name__)


def _fla_gated_delta_step(bound: dict[str, Any]) -> Step:
    """fla's per-token recurrence, for stepping inside a chunk whose intermediates its Triton kernels keep."""
    beta = bound["beta"]
    if bound["use_beta_sigmoid_in_kernel"]:
        beta = torch.sigmoid(beta.float()) * (2.0 if bound["allow_neg_eigval"] else 1.0)
    return _gated_delta_step(bound["k"], bound["v"], bound["g"], beta, bound["use_qk_l2norm_in_kernel"])


def serve_fla_gated_delta_chunk(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """``fla.ops.gated_delta_rule.chunk_gated_delta_rule``: the chunk-state buffer its forward drops.

    ``chunk_gated_delta_rule_fwd_h`` returns ``(h, v_new, final_state)`` with ``h`` the
    state entering each chunk, ``[batch, chunks, heads, K, V]`` in the input dtype,
    and ``final_state`` fp32. It is looked up as a global of the forward's module at
    call time, so the recording stand-in sees exactly what the kernel wrote.
    """
    module = sys.modules[fn.__module__]
    _install_fla_recorder(module)
    _fla_capture.results = captured = []
    try:
        output = fn(*args, **kwargs)
    finally:
        _fla_capture.results = None
    bound = arguments(fn, args, kwargs)
    extra = bound["kwargs"]
    if bound["cu_seqlens"] is not None or bound["cp_context"] is not None:
        return output, Unavailable("kernel sites do not cover fla's packed (cu_seqlens) layout")
    if extra.get("use_gate_in_kernel", False):
        return output, Unavailable(f"{fn.__name__} was called with use_gate_in_kernel; kernel sites cover the log-space gate only")
    if len(captured) != 1:
        return output, Unavailable(
            f"{fn.__name__} ran chunk_gated_delta_rule_fwd_h {len(captured)} times for this call; "
            f"the fla backend that ran keeps its chunk states inside its own kernel"
        )
    h, _, final = captured[0]
    if bound["state_v_first"]:
        h = h.transpose(-1, -2)
        final = None if final is None else final.transpose(-1, -2)
    chunk_size = extra.get("chunk_size", inspect.signature(module.chunk_gated_delta_rule_fwd).parameters["chunk_size"].default)
    return output, recurrence(h, final, chunk_size, bound["k"].shape[1], _walk(_fla_gated_delta_step(bound), chunk_size), fn.__name__)


def serve_fla_gated_delta_recurrent(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """``fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule``: the final state it returns.

    The fused per-token kernel keeps the state in registers; only the final one
    comes back, when asked for. In decoding (one token) that is the token's state.
    """
    output = fn(*args, **kwargs)
    bound = arguments(fn, args, kwargs)
    length = bound["k"].shape[1]
    final = output[1]
    if bound["cu_seqlens"] is not None:
        return output, Unavailable("kernel sites do not cover fla's packed (cu_seqlens) layout")
    if final is None:
        return output, Unavailable(f"{fn.__name__} was called without output_final_state; it exports nothing else")
    if length != 1:
        return output, Unavailable(f"{fn.__name__} exports the final state only; over {length} tokens the per-token states stay in the kernel")
    initial = bound["initial_state"]
    initial = torch.zeros_like(final) if initial is None else initial.to(final.dtype)
    if bound["state_v_first"]:
        initial, final = initial.transpose(-1, -2), final.transpose(-1, -2)
    return output, Recurrence(torch.stack([initial, final], 1), None, 1, length, None, fn.__name__)


#: transformers' hub-kernel name -> serve function for its reference implementation.
REFERENCE: dict[str, Callable] = {
    "chunk_gated_delta_rule": serve_gated_delta_chunk,
    "fused_recurrent_gated_delta_rule": serve_gated_delta_recurrent,
    "chunk_kda": serve_kda_chunk,
    "fused_recurrent_kda": serve_gated_delta_recurrent,
    "mamba_chunk_scan_combined": serve_mamba2_chunk,
    "selective_scan_fn": serve_mamba1_scan,
    "selective_state_update": serve_mamba2_decode,
}

#: (package, hub-kernel name) -> serve function for the library's own kernel.
LIBRARY: dict[tuple[str, str], Callable] = {
    ("fla", "chunk_gated_delta_rule"): serve_fla_gated_delta_chunk,
    ("fla", "fused_recurrent_gated_delta_rule"): serve_fla_gated_delta_recurrent,
}
