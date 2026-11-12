"""Recurrent states from chunked linear-attention and state-space kernels.

Gated delta rule layers (Qwen3-Next, Qwen3.5), Kimi delta attention layers
(Kimi Linear, GLM-5-Next), and Mamba2 scans execute their recurrences in chunks.
Within each chunk, the kernel computes over matrices spanning the chunk's tokens,
and only materializes the recurrent state at chunk boundaries so the next chunk
can consume it. Because no implementation computes the recurrent state at
interior tokens of a chunk during the forward pass, those states cannot be
exported directly. However, the chunk-boundary states and the intra-chunk
intermediates act as sufficient statistics: the state after an interior token
``t`` can be obtained by taking the preceding chunk's boundary state and
applying the kernel's intra-chunk update formula truncated at ``t``.

Each covered mixer module exposes two sites, both returned in fp32:

* ``chunk_states(chunks=, heads=)``: the recurrent state entering each chunk
  (with index ``0`` as the initial state), of shape
  ``[batch, chunks, heads, ...]``, plus the final state after the last chunk as
  one additional entry when the kernel produced it. Served as an
  ``"exact export"`` directly from the kernel's state buffer (upcast to fp32 if
  the kernel stored it in a lower-precision dtype; ``fla`` stores its chunk
  buffer in the input dtype and its final state in fp32, while transformers'
  reference kernels store all states in fp32).
* ``states(tokens=, heads=)``: the recurrent state after each requested token,
  of shape ``[batch, tokens, heads, ...]``. When a token sits at the end of a
  chunk whose following boundary state was exported, that boundary state is
  returned as an ``"exact export"``. For tokens inside a chunk, the state is
  ``"derived"`` by advancing the preceding chunk boundary in fp32 using the
  kernel's intra-chunk formula over its recorded intermediates (pairwise decays,
  keys, and delta-corrected values for the gated delta rule; the same with a
  per-channel decay for Kimi delta attention; segment decays, discretized inputs,
  and ``B`` for Mamba2), using one batched contraction per touched chunk.
  Mamba1's selective scan (Mamba, FalconMamba, Jamba, Zamba) binds the state
  after every token in its loop, so every token is an ``"exact export"``. When a
  state is ``"derived"``, the provenance note verifies the intra-chunk formula
  by running it to the end of each touched chunk and reporting the maximum
  relative difference against the kernel's exported next boundary.

How states and intermediates are captured across kernel implementations:

* Transformers' Python reference kernels (``torch_chunk_gated_delta_rule``,
  ``chunk_kimi_delta_attention``, ``mamba2_chunk_scan``, ``mamba_selective_scan``,
  and the single-token decoding kernels ``torch_recurrent_gated_delta_rule`` and
  ``recurrent_kimi_delta_attention``) assign their boundary states and
  intra-chunk intermediates to local variables. Their serve functions run the
  kernel under the same AST instrumentation used by ``.source``, record those
  local variable bindings, and return the kernel's output without altering the
  arithmetic or operation order.
* The single-token decoding functions ``mamba2_selective_state_update`` and
  ``mamba_selective_state_update`` update the passed-in ``state`` cache tensor
  in place, so their serve function clones ``state`` before and after the call.
* In ``fla``, the chunked kernel computes chunk-boundary states in a Python
  helper (``chunk_gated_delta_rule_fwd_h``) and discards them after forming the
  output. A thin wrapper installed once per process on that helper captures the
  returned chunk states on the active thread. Because ``fla`` keeps its
  intra-chunk intermediates inside Triton kernels, interior token states inside
  an ``fla`` chunk are derived by stepping the preceding boundary token by token
  with the fp32 per-token recurrence. ``fla``'s fused recurrent kernel exports
  only the final state, which during single-token decoding is the only token's
  state.

Each kernel is a row of the catalog in modeling/kernels/transformers.py, which
`TransformersModel` registers when the model is constructed.

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
from torch._subclasses.fake_tensor import FakeTensor

from ...intervention import sites
from ...intervention.sites import Provenance, Sites, Unavailable
from ...intervention.source import arguments, record

#: Epsilon added inside the square root by transformers' and fla's ``l2norm``.
_L2NORM_EPS = 1e-6

#: Signature ``advance(chunk, boundary, offsets)``: advances the fp32
#: ``[batch, heads, ...]`` state entering ``chunk`` to each sorted intra-chunk
#: offset in ``offsets``, returning ``[batch, len(offsets), heads, ...]``.
#: ``None`` when the kernel exports a state after every token.
Advance = Callable[[int, torch.Tensor, list[int]], torch.Tensor] | None

#: Signature ``step(state, t)``: advances an fp32 ``[batch, heads, ...]`` state across token ``t``.
Step = Callable[[torch.Tensor, int], torch.Tensor]


def _states(fn: Callable, args: tuple, kwargs: dict, name: str) -> tuple[Any, list[torch.Tensor]]:
    """Run ``fn`` under AST instrumentation and return its output along with the distinct tensors bound to ``name``."""
    output, bound = record(fn, args, kwargs, (name,))
    return output, bound[name]


class Recurrence(Sites):
    """Site reader for chunk-boundary states and per-token recurrent states.

    ``boundaries`` has shape ``[batch, n, heads, ...]`` and holds the state
    entering each of the first ``n`` chunks of length ``chunk_size`` over a
    sequence of ``length`` tokens, in the dtype used by the kernel. ``final``
    holds the state after the last chunk when the kernel produced it, or ``None``.
    Construct instances via `recurrence`, which validates the boundary count.
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
        """Return the state entering chunk ``index`` (or ``self.chunks`` for the final state) in fp32."""
        if index == self.boundaries.shape[1]:
            return self.final.float()
        return self.boundaries[:, index].float()

    def _stack(self, states: list[torch.Tensor], spec: dict[str, Any], axis: str) -> torch.Tensor:
        """Stack ``states`` into ``[batch, len(states), heads, ...]``, slice along ``heads``, and squeeze integer-indexed axes."""
        shape = self.boundaries.shape
        if states:
            value = torch.stack(states, 1)
        else:
            value = self.boundaries.new_zeros((shape[0], 0, *shape[2:]), dtype=torch.float32)
        value = sites.select(value, 2, sites.indices(spec.get("heads"), shape[2], value.device))
        return sites.squeeze(value, spec, {"heads": 2, axis: 1}).clone()

    def chunk_states(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        sites.check_axes(spec, "chunk_states", ("chunks", "heads"))
        positions = sites.offsets(spec.get("chunks"), self.count)
        value = self._stack([self._boundary(i) for i in positions], spec, "chunks")
        kept = f"kept in {self.boundaries.dtype}" if self.boundaries.dtype is not torch.float32 else "fp32"
        final = f", index {self.chunks} final" if self.count > self.chunks else "; the kernel returned no final state"
        if self.chunk_size == 1:
            note = f"the kernel's own state after each token, {kept}, index 0 initial{final}"
        else:
            note = f"the state entering each chunk of {self.chunk_size} tokens, {kept}, index 0 initial{final}"
        return value, Provenance("exact export", self.op_name, note)

    def states(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        sites.check_axes(spec, "states", ("tokens", "heads"))
        positions = sites.offsets(spec.get("tokens"), self.length)
        sites.warn_if_large((self.boundaries.shape[0], len(positions), *self.boundaries.shape[2:]), "states", "tokens=, heads=")
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
            # Run a single batched contraction for all requested offsets in this
            # chunk. When the kernel exported the next chunk boundary, include the
            # chunk's final offset so we can measure the drift against that boundary.
            offsets = sorted(t - start for t in tokens)
            last = end - 1 - start
            if exported_next and last not in offsets:
                offsets.append(last)
            states = self.advance(chunk, self._boundary(chunk), offsets)
            for i, offset in enumerate(offsets):
                if start + offset in tokens:
                    computed[start + offset] = states[:, i]
            if exported_next and not isinstance(states, FakeTensor):
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
    """Return a `Recurrence` reader if the exported boundary count matches ``length`` and ``chunk_size``, or `Unavailable` otherwise."""
    chunks = -(-length // chunk_size)
    count = boundaries.shape[1] + (final is not None)
    if count not in (chunks, chunks + 1):
        return Unavailable(
            f"{op_name} produced {count} chunk states for {chunks} chunks of {chunk_size} tokens; "
            f"its forward no longer has the shape this reader expects"
        )
    return Recurrence(boundaries, final, chunk_size, length, advance, op_name)


# ---------------------------------------------------------------------------
# Transformers reference kernels: gated delta rule (Qwen3-Next, Qwen3.5),
# Kimi delta attention (Kimi Linear, GLM-5-Next), Mamba2 scan, Mamba1 scan
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + _L2NORM_EPS)


def _chunked(x: torch.Tensor, dim: int, chunk_size: int) -> torch.Tensor:
    """Zero-pad ``x`` along ``dim`` to a multiple of ``chunk_size`` and reshape that axis into ``(chunks, chunk_size)``."""
    length = x.shape[dim]
    pad = (chunk_size - length % chunk_size) % chunk_size
    if pad:
        widths = [0, 0] * (x.dim() - 1 - dim) + [0, pad]
        x = F.pad(x, widths)
    return x.reshape(*x.shape[:dim], -1, chunk_size, *x.shape[dim + 1 :])


def _walk(step: Step, chunk_size: int) -> Advance:
    """Build an `Advance` function that steps the chunk boundary token by token using ``step``."""

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
    """Build the intra-chunk state advance function for the chunked gated delta rule.

    Across a chunk, the kernel forms ``v_new = new_values - k_cumdecay @ S_0``
    (the delta-corrected values given the entering boundary state ``S_0``) and
    computes the next boundary state as
    ``exp(G_last) S_0 + sum_j exp(G_last - G_j) k_j^T v_new_j``, where ``G`` is
    the cumulative decay. Truncating that sum at intra-chunk offset ``p`` gives
    the state after position ``p``:
    ``exp(G_p) S_0 + sum_{j <= p} exp(G_p - G_j) k_j^T v_new_j``. The lower-triangular
    weights ``exp(G_p - G_j)`` for ``j <= p`` are already computed by the kernel as
    ``pairwise_decay``. ``key`` is the kernel's fp32 (and optionally L2-normalized)
    key tensor of shape ``[batch, heads, chunks, chunk, K]``.
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
    """Build the intra-chunk state advance function for Mamba2's chunk scan.

    The kernel computes the next chunk boundary as
    ``exp(A_last) S_0 + sum_j exp(A_last - A_j) (x_j dt_j) B_j^T``, where ``A`` is
    the cumulative discretized decay. Truncating at offset ``p`` replaces
    ``A_last`` with ``A_p`` and restricts the sum to ``j <= p`` using the kernel's
    lower-triangular decay matrix ``L``. ``hidden_states`` is ``x * dt`` and ``B``
    is expanded across heads, both of shape ``[batch, chunks, chunk, heads, ...]``.
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
    """Build the intra-chunk state advance function for Kimi delta attention.

    Kimi delta attention follows the gated delta rule with a per-key-channel
    decay vector ``G`` of shape ``[chunk, K]`` broadcast across the state's value
    columns. The kernel computes the next boundary as
    ``exp(G_last) * S_0 + sum_j (k_j * exp(G_last - G_j))^T v_new_j``. Truncating
    at offset ``p`` uses the lower-triangular channel decay tensor ``decay_mask``
    (``exp(G_p - G_j)`` for ``j <= p``, shape
    ``[batch, heads, chunks, chunk, chunk, K]``). ``key`` has shape
    ``[batch, heads, chunks, chunk, K]`` and ``values`` is the WY-transformed
    value tensor of shape ``[batch, heads, chunks, chunk, V]``.
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
    """Build an fp32 per-token recurrence step for the gated delta rule over ``[batch, tokens, heads, ...]`` inputs.

    Applies ``S_t = S' + k_t (beta_t (v_t - k_t^T S'))^T`` with
    ``S' = S_{t-1} exp(g_t)``, matching ``torch_recurrent_gated_delta_rule``.
    ``beta`` is the gate after any in-kernel sigmoid scaling. When ``key`` has
    fewer heads than ``value`` (grouped value heads in ``fla``), ``key`` is
    repeated across value heads to match the kernel's layout.
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
    """Return `Unavailable` if any recorded intermediate is missing or differs from its expected shape, or ``None`` if all match."""
    for name, (tensor, shape) in expected.items():
        if tensor is None or tuple(tensor.shape) != shape:
            found = "nothing" if tensor is None else f"shape {tuple(tensor.shape)}"
            return Unavailable(f"{fn.__name__} bound {name} as {found}, expected {shape}; its intra-chunk formula no longer has the shape this reader expects")
    return None


def _last(recorded: dict[str, list[torch.Tensor]], name: str) -> torch.Tensor | None:
    return recorded[name][-1] if recorded[name] else None


def serve_gated_delta_chunk(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Serve ``torch_chunk_gated_delta_rule`` by recording its chunk-boundary states and intra-chunk intermediates."""
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
    # The first recorded binding of `key` is the fp32 (and optionally L2-normalized)
    # key of shape [batch, heads, length, K], before the kernel pads and reshapes it into chunks.
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
    """Serve ``torch_recurrent_gated_delta_rule`` and ``recurrent_kimi_delta_attention`` (chunk size 1).

    In both functions, each loop iteration binds ``last_recurrent_state`` twice:
    once after applying the decay and once after adding the delta update. Every
    second binding after the initial state is the state completed for that token.
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
    """Serve ``chunk_kimi_delta_attention`` by recording its chunk-boundary states and intra-chunk intermediates."""
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
    # Use the final bindings of each intermediate: `key` is padded and normalized
    # ([batch, heads, padded, K]), `value` is WY-transformed, and `g` holds the
    # chunked cumulative decay.
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
    """Serve ``mamba_selective_scan`` (Mamba, FalconMamba, Jamba, Zamba).

    The sequential loop binds ``ssm_state`` (shape ``[batch, channels, state_size]``)
    after every token, so each token is treated as a chunk of size 1 and served
    as an ``"exact export"`` with ``heads=`` indexing the channel axis. The
    ``mambapy`` and associative-scan code paths only bind the final state and
    return `Unavailable`.
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
    """Serve ``mamba2_chunk_scan`` by recording its chunk-boundary state tensor and intra-chunk intermediates."""
    output, recorded = record(fn, args, kwargs, ("new_states", "hidden_states", "B", "A_cumsum", "L"))
    states = recorded["new_states"]
    bound = arguments(fn, args, kwargs)
    length, chunk_size = bound["hidden_states"].shape[1], bound["chunk_size"]
    if len(states) != 1:
        return output, Unavailable(f"{fn.__name__} bound new_states {len(states)} times; expected once")
    batch, heads, head_dim, state_dim = states[0].shape[0], states[0].shape[2], states[0].shape[3], states[0].shape[4]
    chunks = -(-length // chunk_size)
    # Before chunking, the final binding of `hidden_states` is `x * dt` and the
    # final binding of `B` is expanded across heads, both of shape [batch, length, heads, ...].
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
    """Serve ``mamba2_selective_state_update`` and ``mamba_selective_state_update`` during single-token decoding.

    Both functions update their ``state`` cache argument in place, so this handler
    clones ``state`` before and after the call to capture the entering and updated states.
    """
    bound = arguments(fn, args, kwargs)
    before = bound["state"].detach().clone()
    output = fn(*args, **kwargs)
    after = bound["state"].detach().clone()
    return output, Recurrence(torch.stack([before, after], 1), None, 1, 1, None, fn.__name__)


# ---------------------------------------------------------------------------
# Flash-Linear-Attention (fla) kernels
# ---------------------------------------------------------------------------

class _Capture(threading.local):
    """Thread-local state whose ``results`` list collects ``chunk_gated_delta_rule_fwd_h`` outputs during `serve_fla_gated_delta_chunk` (``None`` outside a served call)."""

    results: list | None = None


_fla_capture = _Capture()
_fla_installed: set[str] = set()
_fla_lock = threading.Lock()


def _install_fla_recorder(module: ModuleType) -> None:
    """Wrap ``module.chunk_gated_delta_rule_fwd_h`` once so its return value is appended to the active thread's capture list."""
    with _fla_lock:
        if module.__name__ in _fla_installed:
            return
        inner = module.chunk_gated_delta_rule_fwd_h

        def recording(*inner_args: Any, **inner_kwargs: Any) -> Any:
            result = inner(*inner_args, **inner_kwargs)
            captured = _fla_capture.results
            if captured is not None:
                captured.append(result)
            return result

        module.chunk_gated_delta_rule_fwd_h = recording
        _fla_installed.add(module.__name__)


def _fla_gated_delta_step(bound: dict[str, Any]) -> Step:
    """Build the fp32 per-token recurrence step matching `fla`'s chunked gated delta rule parameters."""
    beta = bound["beta"]
    if bound["use_beta_sigmoid_in_kernel"]:
        beta = torch.sigmoid(beta.float()) * (2.0 if bound["allow_neg_eigval"] else 1.0)
    return _gated_delta_step(bound["k"], bound["v"], bound["g"], beta, bound["use_qk_l2norm_in_kernel"])


def serve_fla_gated_delta_chunk(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Serve ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` by capturing its inter-chunk state buffer.

    ``chunk_gated_delta_rule_fwd_h`` returns ``(h, v_new, final_state)``, where
    ``h`` holds the state entering each chunk (shape ``[batch, chunks, heads, K, V]``
    in the input dtype) and ``final_state`` is in fp32. Because the forward pass
    resolves ``chunk_gated_delta_rule_fwd_h`` as a module global at call time,
    the wrapper installed by `_install_fla_recorder` intercepts the buffer before
    the outer function discards it.
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
    """Serve ``fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`` for single-token decoding.

    The fused recurrent Triton kernel keeps intermediate states in registers and
    only writes out ``final_state`` when ``output_final_state=True``. For a
    1-token decode call, the initial state and ``final_state`` give all boundary
    and token states.
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
