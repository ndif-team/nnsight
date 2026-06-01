"""Continuous batching server with vanilla HF inference (no paged attention).

This is the NDIF default backend. It provides true continuous batching —
requests enter and leave the batch dynamically, mixed prefill and decode
in the same step — using a token-budget scheduler with chunked prefill.

Uses standard attention and a single server-owned ``DynamicCache``
(persistent across forward passes). No paged attention, no prefix
sharing, no block allocation. Internal operations are identical to
``model.generate()``.

The server wraps any ``LanguageModel`` externally — users write
interventions against ``LanguageModel`` as usual, and the server
manages batching behind the scenes.

Scheduling (each step)::

    1. Drain new requests from queue into pending list
    2. Schedule under token budget (decode-first, then prefill/chunked):
       - All decoding requests: 1 token each
       - Continuing chunked prefills: up to remaining budget
       - New prefills from pending: up to remaining budget, chunk if needed
    3. Reconcile the persistent cache's rows to the active set, then
       build inputs at ``len(active)`` dense rows (row i = i-th active)
    4. One forward pass with interleaver hooks, mutating the
       persistent cache in place
    5. Sample for decode + completed prefills; skip for partial chunks
    6. Remove finished requests; next step's reconcile compacts the cache

Dense-row / persistent-cache model:
    The server owns one ``DynamicCache`` whose batch dim equals the
    number of *currently active* requests (dense — not padded to
    ``max_batch_size``). A request's row is its position in
    ``self._active``; it is not stored on the request and is recomputed
    each step. HF grows the cache in ``T`` via ``cat`` each forward;
    there is no per-step merge or split.

    The cache's rows are reconciled to the active set only on churn
    (admission/finish) by ``_reconcile_cache`` — a no-op in steady-state
    decode, so the steady path pays no per-step cache copy. On churn,
    surviving rows are copied to their new dense positions and admissions
    get a zero row; when no survivor shares the cache it is dropped so
    ``T`` doesn't grow unbounded across request generations.

    Padding still exists across rows whose ``real_seq_len`` differs
    (HF's ``DynamicCache`` keeps a shared ``T`` across all rows), but it
    is paid once at cache growth, not rebuilt per step. Each request
    carries a ``cache_mask`` recording which of its row's cache
    positions hold real (vs pad) K/V — the invariant
    ``len(cache_mask) == T`` is maintained every step.

The server is an opt-in continuous-batching accelerator for validated
model families (see ``_VANILLA_SUPPORTED_MODEL_TYPES`` /
``_check_model_type_supported``), not a universal backend; unlisted
models fall back to plain ``model.trace`` / ``model.generate``.
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

from ...intervention.errors import capture_deferred
from ...intervention.tracing.globals import Globals
from ..common.request_helper import NNsightRequestHelper

if TYPE_CHECKING:
    from ..language import LanguageModel


logger = logging.getLogger(__name__)


# HF ``config.model_type`` strings for architectures we've validated
# compose with vanilla's batching protocol (decoder-only, returns
# ``DynamicCache`` without internal eviction). The CB server is an
# opt-in accelerator for hot model families, not a universal backend —
# unlisted models fall back to plain ``model.trace`` / ``generate``
# instead of being silently corrupted by a protocol mismatch.
#
# Extend via PR after a parity check against ``model.generate(batch_size>1)``,
# or pass ``extra_allowed_model_types=`` to ``VanillaBatchServer`` to
# admit one without a code change.
_VANILLA_SUPPORTED_MODEL_TYPES: frozenset = frozenset({
    "gpt2",                                  # test fixture
    "llama",                                 # Llama 1/2/3.x, CodeLlama, Tulu, SmolLM2 (model_type=llama)
    "qwen2", "qwen3", "qwen2_moe", "qwen3_moe",
    "mistral",                               # sliding-window enforced via attn mask, no DynamicCache eviction
    "mixtral",                               # MoE, dense K/V per layer
    "gemma", "gemma2",                       # Gemma 2 sliding-window confirmed empirically
    "gpt_neox",                              # GPT-NeoX, Pythia
    "olmo", "olmo2", "olmoe",
    "phi", "phi3",
    "falcon",
    "starcoder2",
    "deepseek_v2", "deepseek_v3",
})


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
    The request's row in the server's persistent ``DynamicCache`` is its
    position in ``self._active`` (dense, reconciled on churn) — not stored
    here. ``cache_mask`` records which positions of that row hold real K/V
    (1) vs pad-token K/V (0).

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
        extra_allowed_model_types: Optional[Any] = None,
    ):
        self.model = model
        self.request_helper = NNsightRequestHelper()
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        # Union the default-supported set with any operator-provided
        # additions. Operators pass an iterable of HF ``model_type``
        # strings they've validated; they cannot remove entries from
        # the default (forking the module-level set is the way to do
        # that — explicit is better than implicit subtraction).
        self.allowed_model_types: frozenset = frozenset(
            _VANILLA_SUPPORTED_MODEL_TYPES
            | (set(extra_allowed_model_types) if extra_allowed_model_types else set())
        )
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

        # Server-owned persistent KV cache. Batch dim equals the number
        # of currently-active requests (dense). HF grows it in T via
        # ``cat`` each forward; the server reconciles its rows to the
        # active set on churn (admission/finish) via ``_reconcile_cache``
        # — a no-op in steady-state decode, so the steady path pays no
        # per-step cache copy. ``_cache_req_order`` records the req_ids
        # the cache currently has rows for, in row order.
        self._persistent_cache: Optional[DynamicCache] = None
        self._cache_req_order: List[str] = []

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        # Refuse to start on models we haven't validated. The CB server
        # is opt-in acceleration for hot families, not a universal
        # backend; unlisted models should use ``model.trace`` /
        # ``model.generate`` directly rather than risk a protocol
        # mismatch silently corrupting output. Idempotent across
        # stop()/start() cycles (just a set membership check).
        self._check_model_type_supported()

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

    def _check_model_type_supported(self) -> None:
        """Refuse to start unless ``model_type`` is in the validated allowlist.

        The CB server is an opt-in accelerator for hot model families,
        not a universal backend. Architectures vary along too many axes
        (attention variants, cache layouts, multimodal inputs,
        encoder-decoder topologies, SSM state schemas, …) to characterise
        "supported" by introspection — any negative test we could write
        would be incomplete in the direction that hurts users
        (silent-corruption false negative). So instead we ship a
        positive list of HF ``model_type`` strings we've validated,
        and everything else falls back to the always-correct slow path
        (``with model.trace(...)`` / ``model.generate(...)``).

        Sources of truth:
          * :data:`_VANILLA_SUPPORTED_MODEL_TYPES` — module-level
            default, grown by PR as new families are validated against
            ``model.generate(batch_size > 1)``.
          * Constructor kwarg ``extra_allowed_model_types`` — operator
            extension for in-house / pre-release models they've
            validated themselves; unioned with the default.

        The runtime invariant in :meth:`_step` is a defense-in-depth
        backstop against drift *within* the allowlist (e.g. a future
        HF release that changes a listed family's cache class, or a
        custom subclass whose internals diverged). The allowlist gates
        entry; the invariant catches drift.
        """
        cfg = self.model._model.config
        model_type = cfg.model_type
        if model_type in self.allowed_model_types:
            return
        model_class = type(self.model._model).__name__
        raise RuntimeError(
            f"VanillaBatchServer does not support this model.\n"
            f"\n"
            f"  Model class: {model_class}\n"
            f"  model_type:  {model_type!r}\n"
            f"\n"
            f"This server is an opt-in continuous-batching accelerator for "
            f"validated model families. Only ``model_type`` strings we've "
            f"checked against ``model.generate(batch_size > 1)`` are admitted; "
            f"unlisted models would risk silent output corruption from "
            f"protocol mismatches (multimodal inputs dropped, encoder pass "
            f"never issued, non-DynamicCache state, …).\n"
            f"\n"
            f"  Currently supported: {sorted(self.allowed_model_types)}\n"
            f"\n"
            f"Options:\n"
            f"  - Use ``with model.trace(...)`` or ``model.generate(...)`` "
            f"directly (no continuous batching) — works for any model HF "
            f"supports.\n"
            f"  - If you've validated this architecture composes with "
            f"vanilla's protocol (decoder-only, returns DynamicCache, no "
            f"internal cache eviction), pass "
            f"``extra_allowed_model_types={{{model_type!r}}}`` to the "
            f"server constructor.\n"
            f"  - For paged/optimized paths: HF paged (NNsightCBManager) "
            f"or vLLM serve."
        )

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

        The request is appended to ``self._active``; its cache row is
        created by the next ``_reconcile_cache`` (a zero row padded to
        the current cache T). The new request's ``cache_mask`` is
        initialised to all-zero with length equal to the current
        persistent-cache T — none of the cache's existing positions hold
        this request's K/V.

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

        T_cache = (
            self._persistent_cache.get_seq_length()
            if self._persistent_cache is not None else 0
        )

        active = ActiveRequest(
            req_id=req.req_id,
            prompt_ids=req.token_ids,
            generated_ids=[],
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
            generation_config=cfg,
            prefilled_len=0,
            cache_mask=[0] * T_cache,
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

        Popping from ``self._active`` leaves a stale row in the
        persistent cache; the next ``_reconcile_cache`` (start of the
        following step) drops it and compacts the survivors.
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

    def _reconcile_cache(self) -> None:
        """Make the persistent cache's rows match the active set, in order.

        No-op when the active set is unchanged since the last step
        (steady-state decode) — that is the whole point of the persistent
        cache: the steady path pays no per-step cache copy. On churn
        (admission and/or finish), the cache is rebuilt once: surviving
        rows are copied to their new dense positions, finished rows are
        dropped, and newly-admitted requests get a zero row padded to the
        current cache length T (their ``cache_mask`` is all-zero for those
        positions, so the pad K/V is masked out of attention).

        Per-layer device placement is preserved: each new zero row is
        allocated on the same device as that layer's existing K/V, so
        ``device_map="auto"`` shards keep working.
        """
        desired = list(self._active.keys())
        if self._persistent_cache is None:
            # First forward will build the cache from scratch (all rows
            # are fresh prefills at T=0); nothing to reconcile.
            self._cache_req_order = desired
            return
        if desired == self._cache_req_order:
            return

        old_pos = {rid: i for i, rid in enumerate(self._cache_req_order)}

        # If no surviving request shares the current cache (e.g. every
        # prior request finished and an all-new batch arrived), drop it
        # entirely so the next forward rebuilds at T=0 — otherwise the
        # cache's T would grow without bound across request batches.
        if not any(rid in old_pos for rid in desired):
            self._persistent_cache = None
            self._cache_req_order = desired
            # The new cache rebuilds at T=0, so no active row has any
            # cached K/V. Their ``cache_mask`` must reset to empty too —
            # otherwise a request admitted while the dropped cache still
            # held T positions keeps a length-T mask that never matches
            # the rebuilt cache, eventually overrunning the per-step
            # attention_mask slice (``attention_mask[row, :len(cm)]``).
            for rid in desired:
                self._active[rid].cache_mask = []
            return

        T = self._persistent_cache.get_seq_length()
        new_cache = DynamicCache()
        for layer in self._persistent_cache.layers:
            K, V = layer.keys, layer.values  # [old_B, H, T, D]
            _, H, _, D = K.shape
            krows, vrows = [], []
            for rid in desired:
                if rid in old_pos:
                    r = old_pos[rid]
                    krows.append(K[r:r + 1])
                    vrows.append(V[r:r + 1])
                else:
                    z = torch.zeros(1, H, T, D, dtype=K.dtype, device=K.device)
                    krows.append(z)
                    vrows.append(torch.zeros_like(z))
            new_layer = DynamicLayer()
            new_layer.update(torch.cat(krows, dim=0), torch.cat(vrows, dim=0))
            new_cache.layers.append(new_layer)

        self._persistent_cache = new_cache
        self._cache_req_order = desired

    def _step(self, scheduled: List[ScheduledItem]):
        """Run one forward pass over all active requests (dense rows).

        The forward batch dim equals the number of active requests; row
        ``i`` is the i-th entry of ``self._active``. Requests scheduled
        this step contribute real input tokens; any active request not
        scheduled (only possible if the token budget is exhausted)
        contributes a pad token whose K/V is masked out of all future
        attention via its ``cache_mask``.

        The persistent cache is reconciled to the active set first (a
        no-op in steady-state decode), then passed to HF as-is and
        reassigned after the forward — HF's ``DynamicCache.update`` cats
        new K/V onto the stored tensors in place; we never merge or split
        rows per step.

        Called via ``_step_with_rollback`` from ``_generation_loop``;
        any exception from this method is caught by the rollback
        wrapper and routed through ``_fail_scheduled`` to scope
        failure to the scheduled batch only.
        """
        model = self.model
        helper = self.request_helper
        device = model.device
        pad_token_id = model.tokenizer.pad_token_id or 0

        # Bring the cache's rows in line with the active set before this
        # step (drops finished rows, appends zero rows for new requests).
        self._reconcile_cache()

        # Dense row order == self._active order == _cache_req_order.
        active_reqs = list(self._active.values())
        B = len(active_reqs)
        row_of = {req.req_id: i for i, req in enumerate(active_reqs)}

        max_input_len = max(item.num_tokens for item in scheduled)
        T_cache = (
            self._persistent_cache.get_seq_length()
            if self._persistent_cache is not None else 0
        )
        T_total = T_cache + max_input_len

        # -- Build inputs at B = num_active rows, dense-indexed --
        input_ids = torch.full(
            (B, max_input_len), pad_token_id,
            dtype=torch.long, device=device,
        )
        attention_mask = torch.zeros(
            (B, T_total), dtype=torch.long, device=device,
        )
        position_ids = torch.zeros(
            (B, max_input_len), dtype=torch.long, device=device,
        )

        # Cache visibility for every active row (carries from prior steps).
        # Each active request's ``cache_mask`` has length T_cache by the
        # invariant maintained at the end of every step.
        for req in active_reqs:
            cm = req.cache_mask
            if cm:
                attention_mask[row_of[req.req_id], :len(cm)] = torch.tensor(
                    cm, dtype=torch.long, device=device,
                )

        # Per-scheduled-row inputs (writes only into rows we're advancing).
        scheduled_by_row: Dict[int, ScheduledItem] = {}
        for item in scheduled:
            req = item.request
            row = row_of[req.req_id]
            n_tok = item.num_tokens
            scheduled_by_row[row] = item

            # Left-pad input_ids
            input_ids[row, max_input_len - n_tok:] = torch.tensor(
                item.token_ids, dtype=torch.long, device=device,
            )
            # Input portion of attention mask
            attention_mask[row, T_cache + max_input_len - n_tok:] = 1
            # Position IDs
            seq_start = req.real_seq_len
            for j in range(n_tok):
                position_ids[row, max_input_len - n_tok + j] = seq_start + j

        # -- Forward pass --
        has_mediators = any(
            helper.mediators.get(item.request.req_id) is not None
            for item in scheduled
        )

        if has_mediators:
            # Row-ordered batch_req_ids so each mediator's
            # ``batch_group = [row, 1]`` matches its row in the
            # ``[B, max_input_len, hidden]`` tensor.
            #
            # ``num_tokens = 1`` per row is intentional for vanilla's
            # padded layout — the base ``Batcher`` narrows on dim 0
            # (per-row), so a per-row slice descriptor is correct
            # regardless of how many prompt tokens a request contributes.
            # vLLM's packed layout and HF paged's ``cu_seq_lens`` layout
            # need real per-request token counts; vanilla is different.
            batch_req_ids = [req.req_id for req in active_reqs]
            num_tokens_map = {rid: 1 for rid in batch_req_ids}
            helper.process_batch_groups(num_tokens_map, batch_req_ids, model)
            model.interleaver.batcher.needs_batching = B > 1
            # The forward tensor has B rows, so the batcher's narrow/swap
            # shape check (``acts.shape[0] == total_batch_size``) must see
            # ``total_batch_size == B``. ``total_batch_size`` is
            # ``sum(last_batch_group)``; force it to span the full batch
            # so every mediator's ``[row, 1]`` group narrows correctly.
            model.interleaver.batcher.last_batch_group = [B - 1, 1]
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
                        past_key_values=self._persistent_cache,
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
                    past_key_values=self._persistent_cache,
                    use_cache=True,
                )

        # Persistent cache was mutated by HF in place. Reassign in case
        # HF returned a different wrapper object (it shouldn't, but the
        # returned reference is authoritative).
        self._persistent_cache = outputs.past_key_values

        # -- Runtime protocol invariant --
        # Architecture-agnostic backstop to the allowlist gate in
        # ``_check_model_type_supported``: catches drift *within* the
        # allowed set — e.g. a future HF release that changes a listed
        # family's cache class, or a custom subclass whose internals
        # diverged. The allowlist controls entry; these two checks
        # guard the contract itself (cache class identity, T grows by
        # exactly max_input_len). Cost: one isinstance + one int
        # compare per step. Fails loudly *before* the cache_mask
        # update would corrupt the invariant.
        if not isinstance(self._persistent_cache, DynamicCache):
            raise RuntimeError(
                f"VanillaBatchServer protocol violation: model returned "
                f"past_key_values of type "
                f"{type(self._persistent_cache).__name__}, but vanilla's "
                f"persistent-cache protocol requires ``DynamicCache``. "
                f"This model substitutes its own cache class internally "
                f"(e.g. ``HybridCache`` / ``MambaCache`` / ``StaticCache``). "
                f"Use ``model.generate(...)`` locally, the HF paged path "
                f"(NNsightCBManager), or vLLM serve instead."
            )
        T_returned = self._persistent_cache.get_seq_length()
        if T_returned != T_cache + max_input_len:
            raise RuntimeError(
                f"VanillaBatchServer protocol violation: persistent cache "
                f"T went from {T_cache} to {T_returned}, expected "
                f"{T_cache + max_input_len} (grew by max_input_len="
                f"{max_input_len}). The model is mutating cache positions "
                f"in ways our protocol doesn't support (eviction, block-"
                f"rounding, sliding-window kv compaction, etc.) — the "
                f"``len(cache_mask) == T`` invariant would break on the "
                f"next step. Use ``model.generate(...)`` locally instead."
            )

        # -- Update cache_mask for ALL active rows --
        # The persistent cache grew by max_input_len positions on every
        # row uniformly (HF cat). For each active row, the new entries
        # are real (1) for its scheduled tokens and pad (0) for the
        # left-padding positions. Non-scheduled active rows get
        # max_input_len pad entries — their K/V at those positions
        # comes from pad_token_id projected through W_k/W_v and must
        # never be attended to.
        for req in active_reqs:
            item = scheduled_by_row.get(row_of[req.req_id])
            n_tok = item.num_tokens if item is not None else 0
            req.cache_mask = (
                req.cache_mask + [0] * (max_input_len - n_tok) + [1] * n_tok
            )

        # -- Detect per-mediator deferred exceptions and finalize
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

        # -- Sample and update state --
        # Gather the scheduled rows out of the [B, vocab] logits into
        # dense scheduled order, then delegate logit transformations
        # (temperature, top_p, top_k, repetition_penalty, ...) to HF's
        # ``LogitsProcessorList`` via ``_sample_next_tokens`` (which
        # indexes purely by scheduled position). This is what makes
        # ``with model.trace(..., temperature=0.7, do_sample=True):``
        # actually stochastic.
        logits_full = outputs.logits[:, -1, :]  # [B, vocab]
        gather_rows = torch.tensor(
            [row_of[item.request.req_id] for item in scheduled],
            dtype=torch.long, device=logits_full.device,
        )
        logits = logits_full.index_select(0, gather_rows)  # dense, scheduled order

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
