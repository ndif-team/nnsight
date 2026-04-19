"""Continuous batching server with vanilla HF inference (no paged attention).

This is the NDIF default backend. It provides true continuous batching —
requests enter and leave the batch dynamically, mixed prefill and decode
in the same step — using a token-budget scheduler with chunked prefill.

Uses standard attention and per-request ``DynamicCache``. No paged
attention, no prefix sharing, no block allocation. Internal operations
are identical to ``model.generate()``.

The server wraps any ``LanguageModel`` externally — users write
interventions against ``LanguageModel`` as usual, and the server
manages batching behind the scenes.

Scheduling (each step)::

    1. Drain new requests from queue into pending list
    2. Schedule under token budget (decode-first, then prefill/chunked):
       - All decoding requests: 1 token each
       - Continuing chunked prefills: up to remaining budget
       - New prefills from pending: up to remaining budget, chunk if needed
    3. Build single padded batch (pad-and-mask for mixed cache lengths)
    4. One forward pass with interleaver hooks
    5. Sample for decode + completed prefills; skip for partial chunks
    6. Remove finished requests, loop

Pad-and-mask strategy:
    KV caches of different lengths are left-padded with zeros and merged
    into a single batched ``DynamicCache``. The attention mask marks
    which cache positions are real (1) vs padding (0). Padding waste is
    bounded by the length difference when requests first merge; after
    that, caches grow in lockstep (1 position per step for decode).
    For short interpretability requests this waste is negligible.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

from ...intervention.batching import Batcher
from ...intervention.tracing.globals import Globals
from ..common.request_helper import NNsightRequestHelper

if TYPE_CHECKING:
    from ..language import LanguageModel


@dataclass
class VanillaRequest:
    """A pending request submitted to the server."""
    req_id: str
    token_ids: List[int]
    gen_kwargs: Dict[str, Any]
    mediator: Any
    trace_id: str
    saved_names: List[str]
    expected_count: int


@dataclass
class ActiveRequest:
    """An in-flight request being generated.

    Tracks both prefill progress (``prefilled_len``) and decode state.
    ``cache_mask`` records which positions in the KV cache are real (1)
    vs padding (0) from batch merging.
    """
    req_id: str
    prompt_ids: List[int]
    generated_ids: List[int]
    max_new_tokens: int
    eos_token_id: int
    past_key_values: Optional[DynamicCache]
    prefilled_len: int = 0
    cache_mask: List[int] = field(default_factory=list)
    finished: bool = False

    @property
    def is_decoding(self) -> bool:
        return self.prefilled_len >= len(self.prompt_ids)

    @property
    def remaining_prompt(self) -> List[int]:
        return self.prompt_ids[self.prefilled_len:]

    @property
    def num_generated(self) -> int:
        return len(self.generated_ids)

    @property
    def real_seq_len(self) -> int:
        """Number of real tokens processed so far (prefilled + generated)."""
        return self.prefilled_len + len(self.generated_ids)


@dataclass
class ScheduledItem:
    """One entry in the scheduled batch for a step."""
    request: ActiveRequest
    num_tokens: int
    is_prefill: bool
    token_ids: List[int]  # actual token IDs to feed this step


class VanillaBatchServer:
    """Continuous batching server using vanilla HF inference.

    Wraps a ``LanguageModel`` externally — the model's public API is
    unchanged. Uses a token-budget scheduler: each step processes at
    most ``token_budget`` tokens across all requests. Decode requests
    (1 token each) are scheduled first, remaining budget goes to
    prefill (chunked if needed).

    All requests run in a single forward pass per step via
    pad-and-mask: KV caches are left-padded to the same length and
    the attention mask handles the rest.

    Args:
        model: A ``LanguageModel`` instance.
        token_budget: Max tokens per step (prefill + decode combined).
        max_batch_size: Max concurrent requests.
    """

    def __init__(
        self,
        model: "LanguageModel",
        token_budget: int = 512,
        max_batch_size: int = 64,
        mediator_timeout: float = 30.0,
    ):
        self.model = model
        self.request_helper = NNsightRequestHelper()
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        # Max seconds a single mediator's worker thread may block the
        # forward pass. Hung user intervention code (infinite loop,
        # blocking I/O) would otherwise wedge the entire batch. The
        # interleaver aborts the offending mediator and continues.
        self.mediator_timeout = mediator_timeout

        self._request_queue: Queue[VanillaRequest] = Queue()
        self._pending: List[VanillaRequest] = []
        self._active: Dict[str, ActiveRequest] = {}
        self._results: Dict[str, dict] = {}
        # Per-request signaling — either a sync Event (for non-async callers)
        # or an asyncio.Future (for FastAPI handlers). Futures are set via
        # loop.call_soon_threadsafe from the background generation thread.
        self._result_signals: Dict[str, Union[threading.Event, asyncio.Future]] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._generation_loop, daemon=True,
            name="vanilla-cb-server",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Build entries from compiled trace
    # ------------------------------------------------------------------

    def build_entries(
        self,
        batched_kwargs: dict,
        mediators: Optional[List[Any]] = None,
    ) -> List[VanillaRequest]:
        """Build request entries from a compiled trace's batched output.

        Extracts per-invoke token IDs from the batched ``input_ids``
        tensor, filters for input mediators (those with ``batch_group``),
        and creates one ``VanillaRequest`` per invoke.

        Args:
            mediators: Explicit mediator list. The HTTP handler passes
                ``tracer.mediators`` because it calls
                ``_setup_interleaver(init_interleaver=False)`` to avoid
                racing with the bg thread on ``model._interleaver``;
                that path cannot rely on ``model._interleaver.mediators``
                being current. When ``None``, falls back to the shared
                interleaver's list (for callers that own it exclusively).
        """
        input_ids = batched_kwargs.get("input_ids")
        attention_mask = batched_kwargs.get("attention_mask")
        max_new_tokens = batched_kwargs.pop("max_new_tokens", 20)

        prompts = []
        if input_ids is not None:
            for i in range(input_ids.shape[0]):
                if attention_mask is not None:
                    mask = attention_mask[i].bool()
                    ids = input_ids[i][mask].tolist()
                else:
                    ids = input_ids[i].tolist()
                prompts.append(ids)

        source_mediators = (
            mediators if mediators is not None else self.model._interleaver.mediators
        )
        input_mediators = [
            m for m in source_mediators
            if m.batch_group is not None
        ]

        saved_names = []
        if input_mediators:
            frame_globals = input_mediators[0].intervention.__globals__
            saved_names = [
                name for name, val in frame_globals.items()
                if id(val) in Globals.saves
            ]

        trace_id = str(uuid.uuid4())
        expected_count = len(input_mediators)

        entries = []
        for idx, mediator in enumerate(input_mediators):
            req_id = f"nns_{trace_id}_{idx}"
            entries.append(VanillaRequest(
                req_id=req_id,
                token_ids=prompts[idx] if idx < len(prompts) else [],
                gen_kwargs={"max_new_tokens": max_new_tokens},
                mediator=mediator,
                trace_id=trace_id,
                saved_names=saved_names,
                expected_count=expected_count,
            ))

        return entries

    # ------------------------------------------------------------------
    # Submit and collect
    # ------------------------------------------------------------------

    def submit(self, request: VanillaRequest) -> threading.Event:
        """Sync submission — returns a threading.Event.

        For callers outside an asyncio event loop (Ray actors, sync tests).
        Blocks the calling thread only; the caller waits via ``event.wait()``.
        """
        event = threading.Event()
        self._result_signals[request.req_id] = event
        self._request_queue.put(request)
        return event

    def submit_async(self, request: VanillaRequest) -> asyncio.Future:
        """Async submission — returns an asyncio.Future bound to the caller's loop.

        The future is resolved by the background generation thread via
        ``loop.call_soon_threadsafe``. Does not block — the caller awaits
        the future with ``await`` or ``asyncio.gather``.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._result_signals[request.req_id] = future
        self._request_queue.put(request)
        return future

    def get_result(self, req_id: str) -> Optional[dict]:
        return self._results.pop(req_id, None)

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    def _generation_loop(self):
        while not self._stop.is_set():
            self._drain_queue()

            if not self._active and not self._pending:
                try:
                    req = self._request_queue.get(timeout=0.1)
                    self._pending.append(req)
                except Empty:
                    continue

            try:
                scheduled = self._schedule()
                if scheduled:
                    self._step(scheduled)
            except Exception as e:
                for req_id in list(self._active.keys()):
                    self._finish_request(req_id, {"__error__": str(e)})

    def _drain_queue(self):
        """Move submitted requests from queue into pending list."""
        while True:
            try:
                req = self._request_queue.get_nowait()
                self._pending.append(req)
            except Empty:
                break

    def _activate_request(self, req: VanillaRequest) -> ActiveRequest:
        """Create an ActiveRequest from a pending VanillaRequest."""
        eos_id = getattr(self.model._model.config, "eos_token_id", None)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        if eos_id is None:
            eos_id = -1

        active = ActiveRequest(
            req_id=req.req_id,
            prompt_ids=req.token_ids,
            generated_ids=[],
            max_new_tokens=req.gen_kwargs.get("max_new_tokens", 20),
            eos_token_id=eos_id,
            past_key_values=DynamicCache(),
            prefilled_len=0,
            cache_mask=[],
        )
        self._active[req.req_id] = active

        self.request_helper.process_new_reqs_direct(
            [(req.req_id, req.mediator, req.trace_id,
              req.saved_names, req.expected_count)],
            self.model,
        )
        return active

    def _finish_request(self, req_id: str, saves: dict):
        """Move a request from active to results and signal the caller.

        Called from the background generation thread. Handles both sync
        threading.Event signals and asyncio.Future signals (the latter
        requires ``call_soon_threadsafe`` because futures can only be
        set from their owning event loop's thread).
        """
        self._active.pop(req_id, None)
        self._results[req_id] = saves
        signal = self._result_signals.pop(req_id, None)
        if signal is None:
            return
        if isinstance(signal, asyncio.Future):
            # Cross-thread: schedule set_result on the owning loop
            if not signal.done():
                signal.get_loop().call_soon_threadsafe(
                    signal.set_result, saves,
                )
        else:
            signal.set()

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _schedule(self) -> List[ScheduledItem]:
        """Schedule requests under the token budget.

        Priority order:
        1. Decode requests (1 token each) — users are already waiting
        2. Continuing chunked prefills — finish what we started
        3. New prefills from pending — chunk if they exceed budget

        Returns:
            List of ``ScheduledItem`` for this step, or empty if nothing
            to do.
        """
        budget = self.token_budget
        scheduled: List[ScheduledItem] = []

        # 1. Decode-first: all active requests that finished prefill
        for req in list(self._active.values()):
            if not req.is_decoding:
                continue
            if budget <= 0:
                break
            token_id = req.generated_ids[-1] if req.generated_ids else req.prompt_ids[-1]
            scheduled.append(ScheduledItem(
                request=req,
                num_tokens=1,
                is_prefill=False,
                token_ids=[token_id],
            ))
            budget -= 1

        # 2. Continuing chunked prefills (active but not fully prefilled)
        for req in list(self._active.values()):
            if req.is_decoding:
                continue
            if budget <= 0:
                break
            remaining = req.remaining_prompt
            chunk_size = min(budget, len(remaining))
            scheduled.append(ScheduledItem(
                request=req,
                num_tokens=chunk_size,
                is_prefill=True,
                token_ids=remaining[:chunk_size],
            ))
            budget -= chunk_size

        # 3. New requests from pending
        admitted = 0
        while budget > 0 and self._pending and len(self._active) + admitted < self.max_batch_size:
            req = self._pending[0]
            active = self._activate_request(req)
            self._pending.pop(0)
            admitted += 1

            chunk_size = min(budget, len(active.prompt_ids))
            scheduled.append(ScheduledItem(
                request=active,
                num_tokens=chunk_size,
                is_prefill=True,
                token_ids=active.prompt_ids[:chunk_size],
            ))
            budget -= chunk_size

        return scheduled

    # ------------------------------------------------------------------
    # Single step: pad-and-mask mixed batch
    # ------------------------------------------------------------------

    def _step(self, scheduled: List[ScheduledItem]):
        """Run one forward pass for all scheduled requests.

        Builds a single padded batch combining prefill and decode
        requests. KV caches are left-padded to the same length;
        the attention mask marks real vs padding positions.
        """
        model = self.model
        helper = self.request_helper
        device = model.device
        pad_token_id = model.tokenizer.pad_token_id or 0

        # -- 1. Compute dimensions --
        max_input_len = max(item.num_tokens for item in scheduled)
        max_cache_len = 0
        for item in scheduled:
            cache_len = item.request.past_key_values.get_seq_length()
            if cache_len > max_cache_len:
                max_cache_len = cache_len
        max_total_len = max_cache_len + max_input_len

        # -- 2. Build input_ids, attention_mask, position_ids --
        batch_size = len(scheduled)
        input_ids = torch.full(
            (batch_size, max_input_len), pad_token_id,
            dtype=torch.long, device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_total_len),
            dtype=torch.long, device=device,
        )
        position_ids = torch.zeros(
            (batch_size, max_input_len),
            dtype=torch.long, device=device,
        )

        for i, item in enumerate(scheduled):
            req = item.request
            n_tok = item.num_tokens

            # Left-pad input_ids
            input_ids[i, max_input_len - n_tok:] = torch.tensor(
                item.token_ids, dtype=torch.long, device=device,
            )

            # Attention mask: [cache_mask_padded | input_mask_padded]
            # Cache portion: left-pad existing cache_mask to max_cache_len
            cm = req.cache_mask
            cache_pad = max_cache_len - len(cm)
            for j, val in enumerate(cm):
                attention_mask[i, cache_pad + j] = val

            # Input portion: left-pad to max_input_len
            input_start = max_cache_len + max_input_len - n_tok
            attention_mask[i, input_start:] = 1

            # Position IDs: based on real sequence position
            seq_start = req.real_seq_len
            for j in range(n_tok):
                position_ids[i, max_input_len - n_tok + j] = seq_start + j

        # -- 3. Merge KV caches --
        past_key_values = self._merge_caches(scheduled, max_cache_len)

        # -- 4. Check if nnsight mediators are active --
        has_mediators = any(
            helper.mediators.get(item.request.req_id) is not None
            for item in scheduled
        )

        # -- 5. Forward pass --
        if has_mediators:
            batch_req_ids = [item.request.req_id for item in scheduled]
            num_tokens_map = {item.request.req_id: 1 for item in scheduled}
            helper.process_batch_groups(num_tokens_map, batch_req_ids, model)
            model._interleaver.batcher.needs_batching = batch_size > 1
            # Apply per-mediator timeout so a hung user intervention
            # can't wedge the shared forward thread.
            model._interleaver.mediator_timeout = self.mediator_timeout
            helper._batch_req_ids = batch_req_ids
            helper._num_scheduled_tokens = num_tokens_map

            Globals.enter()
            try:
                with model._interleaver:
                    outputs = model._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
            finally:
                Globals.exit()
            # NOTE: intentionally skipping handle("result")/check_cache_full/
            # check_dangling_mediators/cancel here. Those finalize the
            # mediator and null interleaver state (batcher, mediators,
            # tracer) — appropriate for a one-shot trace, but wrong for
            # continuous batching where the interleaver is reused across
            # forward passes. Per-request finalization happens below in
            # `helper.finalize_mediators` when max_new_tokens/EOS hits.
        else:
            # Plain forward pass — no nnsight interventions
            with torch.no_grad():
                outputs = model._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

        # -- 6. Split output cache back to per-request --
        self._split_cache(scheduled, outputs.past_key_values, max_cache_len, max_input_len)

        # -- 7. Sample and update state --
        logits = outputs.logits[:, -1, :]  # [batch, vocab] — last position (left-padded)

        finished_ids = set()
        for i, item in enumerate(scheduled):
            req = item.request

            if item.is_prefill:
                req.prefilled_len += item.num_tokens
                if not req.is_decoding:
                    # Chunked — don't sample yet
                    continue
                # Prefill just completed — sample first decode token

            next_token = logits[i].argmax(dim=-1).item()
            req.generated_ids.append(next_token)

            if req.num_generated >= req.max_new_tokens:
                finished_ids.add(req.req_id)
            elif next_token == req.eos_token_id:
                finished_ids.add(req.req_id)

        # -- 8. Finalize finished requests --
        # Collect saves per-request (not across all finished at once) so each
        # client gets its own saves dict. Collating into a single dict
        # would alias same-named variables (e.g. every trace saves
        # ``logits``) and hand the last-writer's value to every caller.
        if finished_ids:
            matched = helper.match_req_ids(finished_ids, strip_suffix=False)
            finished_keys = helper.finalize_mediators(
                matched, finished_ids, model,
            )
            per_req = {}
            for base_id, mediator, internal_key in matched:
                if base_id not in finished_ids:
                    continue
                one_matched = [(base_id, mediator, internal_key)]
                one_keys = {internal_key}
                one_saves, one_removals = helper.collect_saves(
                    one_matched, one_keys,
                )
                helper.cleanup_finished(one_keys, one_removals)
                per_req[base_id] = one_saves

            for req_id in finished_ids:
                self._finish_request(req_id, per_req.get(req_id, {}))

    # ------------------------------------------------------------------
    # Cache merge / split helpers
    # ------------------------------------------------------------------

    def _get_layer_devices(self) -> List[torch.device]:
        """Per-layer devices for the wrapped HF model (cached).

        Handles multi-GPU shards from ``device_map="auto"``. Reads each
        transformer layer's parameter device directly — works whether or
        not ``hf_device_map`` was set, and independent of architecture.
        """
        cached = getattr(self, "_layer_devices_cached", None)
        if cached is not None:
            return cached

        hf_model = self.model._model
        for path in ("model.layers", "transformer.h", "gpt_neox.layers", "transformer.layers"):
            module = hf_model
            found = True
            for part in path.split("."):
                if not hasattr(module, part):
                    found = False
                    break
                module = getattr(module, part)
            if found and hasattr(module, "__len__") and len(module) > 0:
                self._layer_devices_cached = [
                    next(layer.parameters()).device for layer in module
                ]
                return self._layer_devices_cached

        raise RuntimeError(
            "Could not locate transformer layer list on model; "
            "add its path to _get_layer_devices."
        )

    def _merge_caches(
        self,
        scheduled: List[ScheduledItem],
        max_cache_len: int,
    ) -> Optional[DynamicCache]:
        """Merge per-request DynamicCaches into one batched cache.

        Allocates each layer's K/V on that layer's own device (handles
        ``device_map="auto"`` sharding). Left-pads shorter caches with
        zeros to ``max_cache_len``. Returns ``None`` if all caches are
        empty (pure prefill batch).
        """
        if max_cache_len == 0:
            return None

        ref_layer = None
        for item in scheduled:
            c = item.request.past_key_values
            if c.get_seq_length() > 0:
                ref_layer = c.layers[0]
                break
        if ref_layer is None:
            return None

        _, num_heads, _, head_dim = ref_layer.keys.shape
        dtype = ref_layer.keys.dtype
        layer_devices = self._get_layer_devices()
        num_layers = len(layer_devices)
        batch = len(scheduled)

        merged = DynamicCache()
        for layer_idx in range(num_layers):
            dev = layer_devices[layer_idx]
            keys = torch.zeros(batch, num_heads, max_cache_len, head_dim, dtype=dtype, device=dev)
            vals = torch.zeros_like(keys)

            for i, item in enumerate(scheduled):
                req_cache = item.request.past_key_values
                seq_len = req_cache.get_seq_length()
                if seq_len > 0:
                    rk = req_cache.layers[layer_idx].keys
                    rv = req_cache.layers[layer_idx].values
                    keys[i, :, max_cache_len - seq_len:, :] = rk[0].to(dev, non_blocking=True)
                    vals[i, :, max_cache_len - seq_len:, :] = rv[0].to(dev, non_blocking=True)

            layer = DynamicLayer()
            layer.update(keys, vals)
            merged.layers.append(layer)

        return merged

    def _split_cache(
        self,
        scheduled: List[ScheduledItem],
        output_cache: DynamicCache,
        max_cache_len: int,
        max_input_len: int,
    ):
        """Split batched output cache back to per-request caches.

        Each slice stays on its layer's device (the next step's merge
        expects tensors to already be device-correct). ``.contiguous()``
        detaches the per-request handle from the shared batched storage.
        """
        num_layers = len(output_cache.layers)

        for i, item in enumerate(scheduled):
            req = item.request
            n_tok = item.num_tokens

            per_req = DynamicCache()
            for layer_idx in range(num_layers):
                out_layer = output_cache.layers[layer_idx]
                k = out_layer.keys[i:i+1].contiguous()
                v = out_layer.values[i:i+1].contiguous()
                new_layer = DynamicLayer()
                new_layer.update(k, v)
                per_req.layers.append(new_layer)
            req.past_key_values = per_req

            # Update cache_mask: [old_mask_padded_to_max_cache | input_mask_padded]
            old_mask = req.cache_mask
            cache_pad = max_cache_len - len(old_mask)
            new_mask = [0] * cache_pad + old_mask
            input_mask = [0] * (max_input_len - n_tok) + [1] * n_tok
            new_mask.extend(input_mask)
            req.cache_mask = new_mask
