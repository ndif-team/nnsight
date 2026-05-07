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
import logging
import threading
import uuid
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from transformers import DynamicCache, GenerationConfig
from transformers.cache_utils import DynamicLayer
from transformers.generation.logits_process import LogitsProcessorList

from ...intervention.batching import Batcher
from ...intervention.errors import capture_deferred
from ...intervention.tracing.globals import Globals
from ..common.request_helper import NNsightRequestHelper

if TYPE_CHECKING:
    from ..language import LanguageModel


logger = logging.getLogger(__name__)


@dataclass
class VanillaRequest:
    """A pending request submitted to the server.

    Carries a full ``GenerationConfig`` so the server delegates sampling
    (temperature, top_p, top_k, repetition_penalty, ...) and EOS handling
    to HF's own primitives — ``model._get_logits_processor`` and the
    ``LogitsProcessorList`` it returns — rather than reimplementing a
    subset inline. This is how the paged continuous-batching path
    (``NNsightCBProcessor._sample``) already works; vanilla used to
    carry a one-key ``gen_kwargs`` dict and silently ignore everything
    else, which made ``with model.trace(..., temperature=0.7,
    do_sample=True):`` come back deterministic.
    """
    req_id: str
    token_ids: List[int]
    generation_config: GenerationConfig
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

    ``eos_token_ids`` is a set (not a scalar) because HF model configs
    like Llama 3 and Qwen expose ``eos_token_id`` as a list of valid
    termination tokens. Collapsing to the first element (the pre-fix
    behavior) caused generation to run past the real end-of-turn token.
    """
    req_id: str
    prompt_ids: List[int]
    generated_ids: List[int]
    max_new_tokens: int
    eos_token_ids: set
    past_key_values: Optional[DynamicCache]
    generation_config: Optional[GenerationConfig] = None
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
        worker_context: Optional[Callable[[dict], Any]] = None,
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
        # Optional factory that wraps each mediator worker's execution of
        # user code in a per-thread context (e.g. NDIF's import/builtin
        # sandbox). Called as ``worker_context(intervention.__globals__)``
        # inside the worker thread right before ``_intervention(*_args)``.
        # Installed on the interleaver in ``start()``; the bg generation
        # thread itself runs no user code, so it intentionally stays
        # unsandboxed.
        self.worker_context = worker_context

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

        # Cache-compatibility probe: VanillaBatchServer hardcodes
        # ``DynamicCache()`` for every request (see ``_activate_request``),
        # so models that require a different cache class fail somewhere
        # inside the model forward — usually with a cryptic AttributeError
        # or TypeError that doesn't point at the actual cause. Probe once
        # at startup and surface a clear error before any request is
        # served. ``_probed`` flag survives stop()/start() cycles.
        if not getattr(self, "_probed", False):
            self._probe_cache_compatibility()
            self._probed = True

        # Install the worker sandbox on the shared interleaver before the
        # bg thread starts. Set unconditionally so prior ``stop()``/
        # ``start()`` cycles pick up a fresh (possibly None) value rather
        # than inheriting the previous server's policy.
        self.model.interleaver.worker_context = self.worker_context
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._generation_loop, daemon=True,
            name="vanilla-cb-server",
        )
        self._thread.start()

    def _probe_cache_compatibility(self) -> None:
        """Verify the wrapped model accepts ``DynamicCache`` for KV state.

        VanillaBatchServer's batching machinery is built around
        ``DynamicCache``: ``_activate_request`` constructs one per
        request, ``_merge_caches`` stacks them along the batch dim, and
        ``_split_cache`` reverses it after the forward. Models with
        non-standard cache layouts cannot use this path:

        - Mamba / RWKV / SSM-based architectures use
          ``LinearAttentionLayer`` (``conv_states`` / ``recurrent_states``,
          no ``.keys`` / ``.values``).
        - T5 / BART / mBART use ``EncoderDecoderCache`` (separate encoder
          and decoder sub-caches).
        - Some compile-targeted setups use ``StaticCache`` (pre-allocated,
          fixed shape).

        Each of those would crash deep inside the model forward with an
        opaque error. Probing here gives the operator a clear failure
        message naming the model class and the right alternative
        backend.

        Cost: one 1-token forward with ``use_cache=True`` and a fresh
        ``DynamicCache``. Runs once per ``VanillaBatchServer`` instance,
        cached across stop/restart cycles.
        """
        hf_model = self.model._model
        # Route the input to wherever the input embedding is — works
        # under ``device_map="auto"`` shards too, since HF's forward
        # handles cross-device routing internally from there.
        embed = hf_model.get_input_embeddings()
        device = embed.weight.device
        input_ids = torch.tensor([[0]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, 1), dtype=torch.long, device=device)

        def _fail(reason: str, cause: Optional[BaseException] = None) -> "RuntimeError":
            err = RuntimeError(
                f"VanillaBatchServer is incompatible with this model.\n"
                f"\n"
                f"  Model class: {type(hf_model).__name__}\n"
                f"  {reason}\n"
                f"\n"
                f"VanillaBatchServer hardcodes ``DynamicCache`` for "
                f"every request (see _activate_request) AND reads it "
                f"back from ``outputs.past_key_values`` after each "
                f"forward to split caches per-request (see "
                f"_split_cache). It also does not propagate "
                f"multimodal kwargs (pixel_values, image_grid_thw, "
                f"audio inputs, etc.) through batch construction.\n"
                f"\n"
                f"Unsupported architectures:\n"
                f"  - Vision-language models (LLaVA family, Qwen2-VL, "
                f"InternVL, Idefics, PaliGemma, BLIP-2, Phi-Vision, …): "
                f"images would be silently dropped from requests.\n"
                f"  - Mamba / RWKV (LinearAttentionLayer): different "
                f"cache schema entirely.\n"
                f"  - T5 / BART / mBART (EncoderDecoderCache): two "
                f"sub-caches plus a separate encoder pass.\n"
                f"  - Compile-targeted setups (StaticCache).\n"
                f"\n"
                f"Alternatives:\n"
                f"  - HF paged path (NNsightCBManager): if the model "
                f"has a paged-aware HF implementation.\n"
                f"  - vLLM serve: if the model has a paged-aware vLLM "
                f"implementation.\n"
                f"  - Local ``with model.generate(...)`` workflow "
                f"(no serve): works for any model HF supports — "
                f"including VLMs and SSMs — no continuous batching."
            )
            if cause is not None:
                raise err from cause
            raise err

        # VLM detection BEFORE the forward: vision-language models
        # typically use a DynamicCache-using text decoder, so the
        # input-acceptance + roundtrip checks below would pass and
        # let the server start. But vanilla's batch construction
        # doesn't carry pixel_values / image_grid_thw / etc., so
        # real image-bearing requests would be silently text-only —
        # coherent-looking generations that ignored the image. That
        # silent-corruption mode is worse than rejecting at startup.
        # Detect via cheap config inspection and refuse before the
        # forward so the operator's signal is unambiguous.
        if self._is_vision_language_model(hf_model):
            _fail(
                "Probe error: detected vision-language model "
                "(multimodal). The text decoder may be DynamicCache-"
                "compatible, but VanillaBatchServer's batch "
                "construction does not propagate pixel_values, "
                "image_grid_thw, or other multimodal kwargs to the "
                "model forward. Real image-bearing requests would be "
                "silently text-only (model would generate from text "
                "alone), which is silent corruption."
            )

        try:
            with torch.no_grad():
                outputs = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=DynamicCache(),
                    use_cache=True,
                )
        except Exception as e:
            _fail(f"Probe error: {type(e).__name__}: {e}", cause=e)

        # Round-trip check: the model must accept ``DynamicCache`` AND
        # return one on ``outputs.past_key_values``. Mamba and similar
        # SSM-based architectures accept the kwarg silently (the
        # forward signature permits it) but produce their own cache
        # type on the output (``MambaCausalLMOutput.past_key_values``
        # is ``MambaCache``, not ``DynamicCache``). ``_split_cache``
        # would then crash with ``AttributeError`` on layer access.
        # Catch the divergence here.
        returned = getattr(outputs, "past_key_values", None)
        if not isinstance(returned, DynamicCache):
            _fail(
                f"Probe error: model accepted ``past_key_values=DynamicCache()`` "
                f"but returned ``outputs.past_key_values`` of type "
                f"{type(returned).__name__} "
                f"(expected ``DynamicCache``). The model likely uses a "
                f"different cache class internally and ignored the kwarg."
            )

    @staticmethod
    def _is_vision_language_model(hf_model: Any) -> bool:
        """Heuristic detection for vision-language / multimodal models.

        Returns True if the model appears to be multimodal. Used by
        ``_probe_cache_compatibility`` to fail closed before the
        forward — even if the text decoder is ``DynamicCache``-
        compatible (and the probe would otherwise pass), vanilla's
        batch construction doesn't propagate multimodal kwargs, so
        image-bearing requests would silently corrupt.

        Detection signals (any one positive triggers rejection):

        - ``config.vision_config``: the canonical signal across most
          modern VLMs (LLaVA family, Qwen2-VL, InternVL, Idefics,
          PaliGemma, BLIP-2, Phi-3-Vision, DeepSeek-VL2, …).
        - ``config.image_token_id`` / ``config.image_token_index``:
          some configs only carry the image-token marker.
        - ``vision_tower`` / ``vision_model`` attribute on the model
          itself: older LLaVA variants put the tower on the model.

        The check is intentionally permissive (false positives are
        cheap — operator routes to the generate fallback, which is
        correct anyway). False negatives would silently corrupt, so
        the bias is toward over-rejection.
        """
        config = getattr(hf_model, "config", None)
        if config is not None:
            if hasattr(config, "vision_config"):
                return True
            if hasattr(config, "image_token_id") or hasattr(
                config, "image_token_index"
            ):
                return True
        if hasattr(hf_model, "vision_tower") or hasattr(hf_model, "vision_model"):
            return True
        return False

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
                ``tracer.mediators`` because it calls ``_run_user_fn``
                without ``_init_shared_interleaver`` to avoid racing with
                the bg thread on ``model.interleaver``; that path cannot
                rely on ``model.interleaver.mediators`` being current.
                When ``None``, falls back to the shared interleaver's list
                (for callers that own it exclusively).
        """
        input_ids = batched_kwargs.get("input_ids")
        attention_mask = batched_kwargs.get("attention_mask")

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
            mediators if mediators is not None else self.model.interleaver.mediators
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

        # Build one ``GenerationConfig`` for the whole trace. All invokes
        # in a batch share the user's kwargs (trace-level), so one config
        # is attached to every ``VanillaRequest`` below. This is the same
        # pattern ``model.generate()`` uses internally: start from the
        # model default, override with user kwargs, validate.
        #
        # ``_build_generation_config`` pops consumed keys from
        # ``batched_kwargs`` so leftover HF-level kwargs (``input_ids``,
        # ``attention_mask``) stay put for downstream use.
        gen_cfg = self._build_generation_config(batched_kwargs)

        entries = []
        for idx, mediator in enumerate(input_mediators):
            req_id = f"nns_{trace_id}_{idx}"
            entries.append(VanillaRequest(
                req_id=req_id,
                token_ids=prompts[idx] if idx < len(prompts) else [],
                generation_config=gen_cfg,
                mediator=mediator,
                trace_id=trace_id,
                saved_names=saved_names,
                expected_count=expected_count,
            ))

        return entries

    def _build_generation_config(self, batched_kwargs: dict) -> GenerationConfig:
        """Merge user kwargs with the model's default ``GenerationConfig``.

        Fields consumed here (``max_new_tokens``, ``temperature``,
        ``top_p``, ``top_k``, ``do_sample``, ``repetition_penalty``,
        ``eos_token_id``, ``pad_token_id``, ``min_new_tokens``,
        ``no_repeat_ngram_size``, etc.) are popped from
        ``batched_kwargs`` so the caller doesn't forward them again.

        Rejects configurations that continuous batching cannot honor
        (``num_beams > 1``, ``num_return_sequences > 1``) with a clear
        error rather than silently collapsing — the equivalence claim
        to ``model.generate()`` is the whole design contract of the
        vanilla path, and silent divergence breaks it.
        """
        # Start from the model's default generation config. ``from_model_config``
        # is the official entry point; it works with both HF model-config objects
        # and dicts.
        base = GenerationConfig.from_model_config(self.model._model.config)

        # Every attribute on a default-constructed ``GenerationConfig`` is
        # eligible to override. We intersect with what the user passed so we
        # don't shadow a field with ``None`` from an unused kwarg slot.
        default_fields = set(vars(GenerationConfig()).keys())
        overrides = {
            k: batched_kwargs.pop(k) for k in list(batched_kwargs.keys())
            if k in default_fields
        }

        if overrides:
            base.update(**overrides)

        # Reject configurations this server can't represent. Beam search
        # and multi-return-sequence require special batch construction
        # that continuous batching doesn't support — silently collapsing
        # them would diverge from ``model.generate()`` output.
        if (base.num_beams or 1) > 1:
            raise ValueError(
                "VanillaBatchServer does not support beam search (num_beams > 1). "
                "Use local model.generate() for beam search."
            )
        if (base.num_return_sequences or 1) > 1:
            raise ValueError(
                "VanillaBatchServer does not support num_return_sequences > 1. "
                "Each invoke produces a single sequence; launch multiple invokes "
                "for multiple samples."
            )

        # HF's own validator catches contradictions like do_sample=True +
        # temperature=0.0, top_p > 1, etc. Fail fast at admission instead of
        # silently producing wrong output at sampling time.
        base.validate()

        return base

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

            # ``_step_with_rollback`` handles its own per-batch failures
            # via ``_fail_scheduled``. The catch-all here is a "should
            # never fire" safety net — if an exception escapes the
            # rollback wrapper (e.g. bug in ``_fail_scheduled``,
            # unexpected ``_schedule`` failure), log loudly and scope
            # cleanup to whatever batch was visible. Do NOT tank
            # ``self._active`` indiscriminately: siblings that weren't
            # in the failing scheduled batch are innocent until proven
            # guilty. The prior indiscriminate behavior was a latent
            # blast-radius hazard (Problem A of C5).
            scheduled: List[ScheduledItem] = []
            try:
                scheduled = self._schedule()
                if scheduled:
                    self._step_with_rollback(scheduled)
            except Exception as e:
                logger.exception(
                    "Unexpected exception in generation loop — this should "
                    "have been caught inside _step_with_rollback via "
                    "_fail_scheduled. Scoping cleanup to the %d scheduled "
                    "requests visible at failure time.",
                    len(scheduled),
                )
                if scheduled:
                    self._fail_scheduled(scheduled, e)

    def _drain_queue(self):
        """Move submitted requests from queue into pending list."""
        while True:
            try:
                req = self._request_queue.get_nowait()
                self._pending.append(req)
            except Empty:
                break

    def _activate_request(self, req: VanillaRequest) -> ActiveRequest:
        """Create an ActiveRequest from a pending VanillaRequest.

        EOS tokens flow from the request's ``generation_config`` first
        (so per-trace overrides take effect) and fall back to the model
        config. Carried as a set to honor multi-EOS configs (Llama 3
        lists ``[128001, 128008, 128009]``; collapsing to index 0
        skips the real end-of-turn token).
        """
        cfg = req.generation_config
        eos = cfg.eos_token_id if cfg is not None and cfg.eos_token_id is not None \
              else getattr(self.model._model.config, "eos_token_id", None)
        if isinstance(eos, int):
            eos_ids = {eos}
        elif isinstance(eos, list):
            eos_ids = set(eos)
        else:
            eos_ids = set()

        max_new_tokens = (cfg.max_new_tokens if cfg is not None else None) or 20

        active = ActiveRequest(
            req_id=req.req_id,
            prompt_ids=req.token_ids,
            generated_ids=[],
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
            generation_config=cfg,
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

        Cross-thread error handling (I7): both ``call_soon_threadsafe``
        and the eventual ``set_result`` can raise — closed loop after
        client disconnect raises ``RuntimeError`` synchronously here;
        a future that became done in a parallel path between our
        ``done()`` check and the loop callback running raises
        ``InvalidStateError`` inside that callback. Either raise must
        be swallowed: if it propagated to ``_generation_loop``'s
        catch-all, a single client disconnect would tank every
        co-batched request.
        """
        self._active.pop(req_id, None)
        self._results[req_id] = saves
        signal = self._result_signals.pop(req_id, None)
        if signal is None:
            return
        if isinstance(signal, asyncio.Future):
            if signal.done():
                # Handler already cancelled / completed via another
                # path; nothing to signal.
                return

            def _safe_set_result(fut=signal, value=saves, _req=req_id):
                try:
                    fut.set_result(value)
                except asyncio.InvalidStateError:
                    # Future completed/cancelled between our ``done()``
                    # check (bg thread) and this callback running on
                    # the loop thread. Handler already gone or already
                    # gave up — fine.
                    logger.debug(
                        "set_result on already-done future for req %s",
                        _req,
                    )

            try:
                signal.get_loop().call_soon_threadsafe(_safe_set_result)
            except RuntimeError:
                # Loop is closed (handler cancelled, client
                # disconnected, process shutting down). Cannot
                # signal — log and move on. Critical: do NOT raise
                # into the bg generation thread.
                logger.warning(
                    "Cannot signal req %s: asyncio loop is closed",
                    req_id,
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

    def _step_with_rollback(self, scheduled: List[ScheduledItem]) -> None:
        """Run ``_step`` with per-batch failure scoping.

        On any exception, invoke ``_fail_scheduled`` to:
        - finalize the scheduled batch with ``__error__`` (and ONLY
          the scheduled batch — innocent siblings in ``_active`` are
          untouched).
        - reset per-step shared state (``_interleaver.mediators``,
          ``helper._batch_req_ids``, etc.) so the next step starts clean.
        - drop helper.mediators entries for the scheduled req_ids so
          they don't linger as orphans (pre-fix memory leak: each
          failed request left a stale dict entry until process exit).

        This is the entry point called by ``_generation_loop``.
        Tests that patch ``_step`` directly exercise the inner method;
        the rollback wrapping is applied here.
        """
        try:
            self._step(scheduled)
        except Exception as e:
            self._fail_scheduled(scheduled, e)

    def _fail_scheduled(
        self, scheduled: List[ScheduledItem], exc: Exception,
    ) -> None:
        """Finalize ONLY the scheduled batch with ``__error__`` and reset
        per-step shared state.

        Scope is the scheduled batch, not ``self._active``. Requests
        currently in ``_active`` but not in ``scheduled`` are untouched
        — the failure didn't affect them.

        Idempotent: safe to call from both ``_step_with_rollback``'s
        inner except AND from ``_generation_loop``'s outer safety-net
        catch-all. Double-invocation for the same batch is a no-op
        because ``_finish_request`` pops from ``_active`` on first
        call (so the second call's state checks see nothing to do).
        """
        import traceback as _tb

        err_envelope = {
            "type_name": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                _tb.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "is_control_flow": False,
        }

        helper = self.request_helper
        model = self.model
        scheduled_ids = {item.request.req_id for item in scheduled}

        # Cancel stranded mediator workers for the scheduled batch.
        # ``Interleaver.__enter__`` started a worker thread per mediator
        # before forward; if forward raised mid-way, those workers are
        # stuck waiting on condition variables. Cancel so they unwind.
        # Best-effort: cancel itself failing shouldn't block finalization.
        for req_id in scheduled_ids:
            med = helper.mediators.get(req_id)
            if med is not None:
                try:
                    med.cancel()
                except Exception:
                    logger.exception(
                        "Failed to cancel mediator for req_id=%s during "
                        "batch failure cleanup; continuing.", req_id,
                    )

        # Clean helper state for the scheduled requests using the same
        # finalize-cleanup sequence as the normal finish path (see
        # ``_step``'s per-request failure finalization). If cleanup
        # itself fails (cleanup inside a failure handler is risky),
        # fall back to directly dropping the helper.mediators entries
        # so we don't leak memory.
        try:
            matched = helper.match_req_ids(scheduled_ids, strip_suffix=False)
            finished_keys = helper.finalize_mediators(
                matched, scheduled_ids, model,
            )
            _, removals = helper.collect_saves(matched, finished_keys)
            helper.cleanup_finished(finished_keys, removals)
        except Exception:
            logger.exception(
                "helper cleanup failed during batch failure handling; "
                "falling back to direct mediator-dict removal to avoid "
                "leak. req_ids=%s", scheduled_ids,
            )
            for req_id in scheduled_ids:
                helper.mediators.pop(req_id, None)

        # Reset per-step shared state on the interleaver. The next
        # ``_step`` call's ``process_batch_groups`` rebuilds all of
        # these, so resetting here is defensive (clean slate) rather
        # than strictly required — but it also drops references so
        # GC can reclaim the failed step's mediators immediately.
        model.interleaver.mediators = []
        model.interleaver.batcher.last_batch_group = None
        model.interleaver.batcher.needs_batching = False
        helper._batch_req_ids = []
        helper._num_scheduled_tokens = {}

        # Finalize per-request with the error envelope. After this,
        # each request is popped from ``_active`` and its caller's
        # signal is set (see ``_finish_request``).
        for req_id in scheduled_ids:
            entry = dict(err_envelope)
            entry["req_id"] = req_id
            self._finish_request(req_id, {"__error__": entry})

    def _step(self, scheduled: List[ScheduledItem]):
        """Run one forward pass for all scheduled requests.

        Builds a single padded batch combining prefill and decode
        requests. KV caches are left-padded to the same length;
        the attention mask marks real vs padding positions.

        Called via ``_step_with_rollback`` from ``_generation_loop``;
        any exception from this method is caught by the rollback
        wrapper and routed through ``_fail_scheduled`` to scope
        failure to the scheduled batch only.
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
            # ``num_tokens_map = {req_id: 1}`` is intentional for vanilla's
            # ``[batch_size, max_input_len, hidden]`` padded layout — NOT a
            # bug. The base ``Batcher`` narrows on dim 0 (per-row), so
            # ``mediator.batch_group = [row_idx, 1]`` is the correct slice
            # descriptor regardless of how many prompt tokens a request
            # contributes. Changing to ``item.num_tokens`` would break
            # vanilla because ``total_batch_size = sum(last_batch_group)``
            # would no longer equal ``acts.shape[0]`` and the narrow
            # check (``_narrow``: ``if acts.shape[0] == total_batch_size``)
            # would fail silently, returning un-narrowed full-batch tensors.
            #
            # vLLM's packed ``[total_tokens, hidden]`` layout and HF paged's
            # ``cu_seq_lens``-indexed layout DO need real per-request token
            # counts, and those paths supply them correctly
            # (``GPUModelRunner`` reads ``scheduler_output.num_scheduled_tokens``;
            # ``NNsightCBManager`` reads ``cu_seq_lens``). Vanilla is
            # structurally different and the hardcoded 1 is the right value.
            num_tokens_map = {item.request.req_id: 1 for item in scheduled}
            helper.process_batch_groups(num_tokens_map, batch_req_ids, model)
            model.interleaver.batcher.needs_batching = batch_size > 1
            # Apply per-mediator timeout so a hung user intervention
            # can't wedge the shared forward thread.
            model.interleaver.mediator_timeout = self.mediator_timeout

            # Per-request error isolation: read each mediator's
            # ``_deferred_exception`` after the forward pass and finalize
            # only the failing requests below. Raising from ``__exit__``
            # would escape ``_step`` and trip ``_generation_loop``'s
            # catch-all, tanking every co-batched sibling.
            model.interleaver.defer_exceptions = True
            try:
                with model.interleaver:
                    outputs = model._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
            finally:
                model.interleaver.defer_exceptions = False
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

        # -- 6b. Detect per-mediator deferred exceptions and finalize
        # those requests with ``__error__``. Skips them in the sampling
        # loop below so we don't append a generated token to a request
        # the user code never finished setting up.
        #
        # ``__error__`` is a full ``DeferredError`` dict (type_name,
        # message, traceback, is_control_flow) — see
        # ``intervention.errors.capture_deferred``. The HTTP handler
        # routes it into the response envelope's ``errors`` list, and
        # the client re-raises via ``surface_server_errors``.
        failed_req_ids: Dict[str, Dict[str, Any]] = {}
        for item in scheduled:
            req_id = item.request.req_id
            med = helper.mediators.get(req_id)
            if med is None:
                continue
            entry = capture_deferred(med, req_id=req_id)
            if entry is not None:
                failed_req_ids[req_id] = entry
                # Clear so a re-used mediator doesn't re-trigger.
                med.deferred_exception = None
                med._deferred_type_name = None
                med._deferred_traceback = None
                med._deferred_is_control_flow = False

        # -- 7. Sample and update state --
        # Delegate logit transformations (temperature, top_p, top_k,
        # repetition_penalty, ...) to HF's ``LogitsProcessorList``, built
        # from each request's ``GenerationConfig`` via the same helper
        # ``model.generate()`` uses. This is what makes ``with
        # model.trace(..., temperature=0.7, do_sample=True):`` actually
        # stochastic — vanilla used to hand-roll a bare argmax that
        # silently ignored every sampling kwarg.
        logits = outputs.logits[:, -1, :]  # [batch, vocab] — last position (left-padded)

        next_tokens = self._sample_next_tokens(logits, scheduled, input_ids)

        finished_ids = set()
        for i, item in enumerate(scheduled):
            req = item.request
            if req.req_id in failed_req_ids:
                continue

            if item.is_prefill:
                req.prefilled_len += item.num_tokens
                if not req.is_decoding:
                    # Chunked — don't sample yet
                    continue
                # Prefill just completed — sample first decode token

            next_token = int(next_tokens[i].item())
            req.generated_ids.append(next_token)

            if req.num_generated >= req.max_new_tokens:
                finished_ids.add(req.req_id)
            elif next_token in req.eos_token_ids:
                finished_ids.add(req.req_id)

        # -- 7b. Finalize failed requests with __error__ --
        for req_id, err_entry in failed_req_ids.items():
            # Clean up helper state (matches normal finalize path) so the
            # mediator doesn't linger in ``helper.mediators``.
            matched = helper.match_req_ids({req_id}, strip_suffix=False)
            finished_keys = helper.finalize_mediators(matched, {req_id}, model)
            _, removals = helper.collect_saves(matched, finished_keys)
            helper.cleanup_finished(finished_keys, removals)
            self._finish_request(req_id, {"__error__": err_entry})

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
    # Sampling
    # ------------------------------------------------------------------

    def _sample_next_tokens(
        self,
        logits: torch.Tensor,
        scheduled: List["ScheduledItem"],
        step_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply HF's ``LogitsProcessorList`` then sample per request's config.

        One ``LogitsProcessorList`` is built per unique ``GenerationConfig``
        in the batch (in practice all requests in a trace share the same
        config, so this is usually built once). History-aware processors
        (``RepetitionPenaltyLogitsProcessor``,
        ``NoRepeatNGramLogitsProcessor``) need the per-request
        ``input_ids`` history of shape ``[batch, seq]``; we reconstruct it
        from ``prompt_ids + generated_ids`` and left-pad to a common
        length so a single processor call covers the whole batch.

        For ``do_sample=True``, samples via ``multinomial``. Otherwise
        argmax. Matches what ``model.generate()`` does after the logits
        processor stage.

        Returns:
            ``[batch]`` long tensor of next-token ids, aligned with the
            rows of ``logits`` / ``scheduled``.
        """
        batch_size = logits.shape[0]
        device = logits.device
        dtype = torch.long

        # Group requests by the id of their GenerationConfig so multiple
        # configs in one step (rare — would require heterogeneous kwargs
        # across invokes of one trace) are handled correctly.
        cfg_groups: Dict[int, tuple] = {}
        for i, item in enumerate(scheduled):
            cfg = item.request.generation_config
            key = id(cfg) if cfg is not None else 0
            if key not in cfg_groups:
                cfg_groups[key] = (cfg, [])
            cfg_groups[key][1].append(i)

        next_tokens = torch.empty(batch_size, dtype=dtype, device=device)

        # Build a padded [batch, max_hist] input_ids tensor covering
        # every request's full history. The pad token is arbitrary for
        # the processors that care (they look at the last real tokens);
        # repetition_penalty scans the whole sequence, so padding with
        # pad_token_id is safer than 0 to avoid penalizing token 0.
        pad_id = self.model._model.config.pad_token_id
        if pad_id is None:
            pad_id = self.model._model.config.eos_token_id
            if isinstance(pad_id, list):
                pad_id = pad_id[0]
            if pad_id is None:
                pad_id = 0

        histories = [
            item.request.prompt_ids + item.request.generated_ids
            for item in scheduled
        ]
        max_hist = max((len(h) for h in histories), default=1)
        max_hist = max(max_hist, 1)
        history_ids = torch.full(
            (batch_size, max_hist), int(pad_id),
            dtype=dtype, device=device,
        )
        for i, h in enumerate(histories):
            if h:
                history_ids[i, -len(h):] = torch.tensor(h, dtype=dtype, device=device)

        for cfg, indices in cfg_groups.values():
            if cfg is None:
                # No config attached (test plumbing or legacy caller) —
                # fall back to argmax so tests that bypass build_entries
                # keep working. The HTTP path always attaches a config.
                next_tokens[indices] = logits[indices].argmax(dim=-1)
                continue

            rows = torch.tensor(indices, dtype=torch.long, device=device)
            row_logits = logits.index_select(0, rows)
            row_history = history_ids.index_select(0, rows)

            processor = self.model._model._get_logits_processor(
                generation_config=cfg,
                input_ids_seq_length=row_history.shape[1],
                encoder_input_ids=None,
                prefix_allowed_tokens_fn=None,
                logits_processor=None,
                device=device,
            )

            processed = processor(row_history, row_logits)

            if cfg.do_sample:
                probs = torch.softmax(processed, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                sampled = processed.argmax(dim=-1)

            next_tokens.index_copy_(0, rows, sampled.to(dtype))

        return next_tokens

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
