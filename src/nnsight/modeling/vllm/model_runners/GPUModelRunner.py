import pickle
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import zstandard as _zstd

_ZSTD_COMPRESSOR = _zstd.ZstdCompressor(level=1)

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from ....intervention.serialization import load
from ....intervention.tracing.globals import Globals
from ..batching import VLLMBatcher
from ..lazy_remote_tensor import strip_lazy

if TYPE_CHECKING:
    from ..vllm import VLLM
else:
    VLLM = Any

if TYPE_CHECKING:

    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput


class NNsightGPUModelRunner(GPUModelRunner):
    """Custom vLLM GPU model runner that interleaves NNsight interventions with model execution.

    Wraps the model with an NNsight :class:`Envoy`, deserializes
    mediators from incoming :class:`NNsightSamplingParams`, and manages
    batch group mappings so each invoke's intervention code sees the
    correct slice of the batch.

    When :data:`pipeline_parallel_size` > 1 the runner also owns the PP
    plumbing: shared per-module metadata, the per-rank ``pp_hook_buffer``,
    the cross-rank :class:`~..pp_listener.PPListener` thread, and the
    per-step readiness gate that waits for every mediator to park at a
    local-module access before firing forward-pass hooks.
    """

    class NNsightRequestHelper:
        """
        Helper class for batching requests in the GPUModelRunner.

        Attributes:
            req_id_to_batch_group_idx: req_id → batch group index
            mediators: req_id → :class:`Mediator`
            trace_contexts: trace_id → context dict (canonical_globals, saved_names, pending_req_ids)
        """

        def __init__(self):

            self.req_id_to_batch_group_idx: Dict[str, int] = {}
            self.mediators: Dict[str, Any] = {}
            self.trace_contexts: Dict[str, dict] = {}

        def _pp_aware_load(self, data: bytes, model: VLLM):
            """Deserialize a mediator with PP-aware persistent ID resolution.

            When PP is enabled the serialized mediator references full-model
            module paths (e.g. ``model.transformer.h.6.ln_1``). On PP
            workers, layers on other stages are :class:`PPMissingLayer`
            stubs with no children, so a direct lookup fails. This unpickler
            walks up the dotted path until it finds an ancestor
            :class:`PPMissingLayer` and returns that — far enough up the
            tree that one of the lazy-tensor-returning Envoys is at the
            access path the user wrote.
            """
            import io

            persistent_objects = model._remoteable_persistent_objects()
            pp_enabled = get_pp_group().world_size > 1

            if not pp_enabled:
                return load(data, persistent_objects)

            from ..pp import is_pp_missing
            from ....intervention.serialization import CustomCloudUnpickler

            class _PPUnpickler(CustomCloudUnpickler):
                def persistent_load(self, pid):
                    if pid in self.persistent_objects:
                        return self.persistent_objects[pid]
                    if isinstance(pid, str) and pid.startswith("Module:"):
                        path = pid[len("Module:"):]
                        parts = path.split(".")
                        for i in range(len(parts) - 1, 0, -1):
                            ancestor_pid = "Module:" + ".".join(parts[:i])
                            if ancestor_pid in self.persistent_objects:
                                ancestor = self.persistent_objects[ancestor_pid]
                                if is_pp_missing(ancestor):
                                    return ancestor
                    raise pickle.UnpicklingError(
                        f"Unknown persistent id: {pid}"
                    )

            return _PPUnpickler(io.BytesIO(data), persistent_objects).load()

        def process_new_reqs(
            self, new_reqs: List["NewRequestData"], model: VLLM
        ) -> None:
            """
            Process new requests and organize them into batch groups for execution.

            Each request carries its own serialized mediator. When multiple
            mediators belong to the same trace (identified by trace_id), the
            first arrival's ``__globals__`` become the canonical reference.
            Subsequent arrivals graft the saved variable entries from the
            canonical globals into their own ``__globals__``, so all mediators
            share the same Python objects for cross-invoke state.
            """

            for new_req in new_reqs:

                extra_args = getattr(new_req.sampling_params, "extra_args", None)
                if not extra_args:
                    continue

                trace_id = extra_args.get("nnsight_trace_id")
                if trace_id is None:
                    continue

                mediator = self._pp_aware_load(
                    extra_args["nnsight_mediator"], model,
                )

                saved_names = extra_args.get("nnsight_saved_names", [])

                if trace_id not in self.trace_contexts:
                    canonical_globals = mediator.intervention.__globals__

                    for name in saved_names:
                        if name in canonical_globals:
                            Globals.saves.add(id(canonical_globals[name]))

                    self.trace_contexts[trace_id] = {
                        "saved_names": saved_names,
                        "canonical_globals": canonical_globals,
                        "expected_count": extra_args.get("nnsight_expected_count", 1),
                        "received_count": 0,
                        "pending_req_ids": set(),
                    }
                else:
                    ctx = self.trace_contexts[trace_id]
                    canonical = ctx["canonical_globals"]
                    med_globals = mediator.intervention.__globals__
                    for name in saved_names:
                        if name in canonical:
                            med_globals[name] = canonical[name]

                ctx = self.trace_contexts[trace_id]

                # Tag the mediator with its stable cross-rank request id
                # BEFORE ``mediator.start(...)`` — the worker thread spawned
                # by ``start()`` may immediately run user code that hits
                # the pp_eproperty short-circuit and captures ``pp_req_id``
                # into the pull closure. Setting it after ``start()`` races:
                # the closure may capture ``None`` and cross-rank pulls fail
                # to look up the composite ``(provider, req_id)`` key.
                mediator.pp_req_id = new_req.req_id

                # Reset the iteration gate for the new request so mediators
                # are not blocked by a previous stop signal.
                interleaver = model.interleaver
                if getattr(interleaver, "_generation_done", False):
                    interleaver._generation_done = False

                mediator.idx = len(interleaver.mediators)
                interleaver.mediators.append(mediator)
                mediator.start(interleaver)

                self.mediators[new_req.req_id] = mediator
                ctx["pending_req_ids"].add(new_req.req_id)
                ctx["received_count"] += 1

        def unflatten(self, model: VLLM):
            """Re-assign batch groups from token-level to prompt-level.

            After the forward pass, logits have one row per *scheduled
            request* (in ``batch_req_ids`` order).  We must walk the
            same ordering used by ``process_batch_groups`` so that each
            mediator's prompt-level index matches its row in the logits
            tensor — even when the batch contains non-NNsight requests
            or requests whose mediators have already finished.
            """

            batch_start = 0
            mediator_set = {id(m) for m in model.interleaver.mediators}

            for req_id in self._batch_req_ids:
                if self._num_scheduled_tokens.get(req_id) is None:
                    continue

                mediator = self.mediators.get(req_id)

                if mediator is None or id(mediator) not in mediator_set:
                    batch_start += 1
                    continue

                mediator.batch_group = [batch_start, 1]
                batch_start += 1
                model.interleaver.batcher.last_batch_group = mediator.batch_group

        def process_batch_groups(
            self,
            num_tokens_scheduled: Dict[str, int],
            batch_req_ids: List[str],
            model: VLLM,
        ) -> None:

            # Clear batch_group for all registered mediators first. Persistent
            # cache hooks read mediator.batch_group live on each forward pass,
            # so a mediator whose request isn't scheduled in this step must
            # report "None" rather than the stale value from an earlier step
            # (which would point out-of-range in the smaller current batch).
            for m in self.mediators.values():
                m.batch_group = None
                # pp_num_tokens is the authoritative per-request token count
                # the cross-rank PP pull sizes its transfer from. Cleared here
                # alongside batch_group so a mediator not scheduled this step
                # never serves a stale count; re-set below for scheduled ones.
                m.pp_num_tokens = None

            batch_start = 0

            mediators = []

            for req_id in batch_req_ids:

                num_tokens = num_tokens_scheduled.get(req_id)
                if num_tokens is None:
                    continue

                mediator = self.mediators.get(req_id)

                if mediator is None:
                    batch_start += num_tokens
                    continue

                mediators.append(mediator)
                mediator.batch_group = [batch_start, num_tokens]
                # Authoritative token count for the PP pull. Unlike
                # batch_group (which unflatten later rewrites to the
                # prompt-level [start, 1] logits view), this is set once per
                # step and never rewritten, so it always equals the producer's
                # buffered leading dim. See pp_envoy._pp_lazy_access and
                # tests/test_pp_num_tokens_unflatten.py.
                mediator.pp_num_tokens = num_tokens

                # Gate-only: count forwards that actually process this request
                # (immune to pipeline bubbles). The readiness gate uses
                # ``count - 1`` as the iteration THIS forward is for; the worker
                # never waits on it.
                mediator._pp_scheduled_count = (
                    getattr(mediator, "_pp_scheduled_count", 0) + 1
                )

                batch_start += num_tokens

            if mediators:
                model.interleaver.batcher.last_batch_group = mediators[-1].batch_group
            else:
                model.interleaver.batcher.last_batch_group = None

            model.interleaver.mediators = mediators

        def match_req_ids(self, req_id_set: set) -> List[tuple]:
            """Match engine-reported request IDs to stored mediators.

            vLLM appends a hash suffix to request IDs (e.g. ``"0-abc123"``
            or ``"uuid-abc123"``). Strip the suffix with ``rsplit`` and
            fall back to an exact match.
            """
            matched = []
            for req_id, mediator in self.mediators.items():
                base_id = req_id.rsplit("-", 1)[0]
                if base_id in req_id_set:
                    matched.append((base_id, mediator, req_id))
                elif req_id in req_id_set:
                    matched.append((req_id, mediator, req_id))
            return matched

        def finalize_mediators(self, matched, finished_req_id_set, model: VLLM) -> set:
            """Run result handler and cancel finished mediators."""
            finished_internal_keys = set()
            for base_id, mediator, internal_key in matched:
                if base_id not in finished_req_id_set:
                    continue

                finished_internal_keys.add(internal_key)

                if mediator.alive:
                    model.interleaver.mediators = [mediator]
                    mediator.batch_group = None
                    with model.interleaver:
                        model.interleaver.handle("result", [base_id])
                        mediator.cancel()
                        model.interleaver.handle()
                # Always remove persistent cache hooks when the request
                # finishes — even if the mediator thread died early
                # (e.g. intervention code was just tracer.cache(); nns.save(c)
                # with no blocking access). Otherwise hooks pile up on the
                # module and keep firing with stale batch_groups from dead
                # mediators.
                mediator.remove_hooks()

            return finished_internal_keys

        def collect_saves(self, matched, finished_internal_keys: set) -> tuple:
            """Collect saved values from mediator frames, namespaced per request.

            Gathers per-invoke saves from frame locals and trace-shared
            saves from canonical globals (only when a trace is fully done).

            Cross-stage handling: each saved value may hold tensors owned by
            other PP ranks (as :class:`LazyRemoteTensor`). :func:`strip_lazy`
            replaces those with the ``NOT_ON_THIS_RANK`` sentinel and reports
            whether this rank owns any real data. A value owned *entirely*
            elsewhere is skipped (its owner ships it); a partially-owned
            value (e.g. a list of activations split across stages) is shipped
            with sentinels in the foreign slots, and the engine merges the
            per-rank contributions position-wise (:func:`merge_saved`).

            Returns:
                ``(saves_by_req, removals)`` —
                ``saves_by_req`` is ``{base_id: {var_name: value}}`` so
                concurrent requests whose user code uses the same
                variable name (``logits``, ``x``, …) don't collide at
                the outer flat-dict layer. The caller (engine / server)
                routes each sub-dict to the matching request output.
                ``removals`` is a list of ``id`` values to discard from
                ``Globals.saves`` after collection.
            """
            saves_by_req: dict = {}
            removals = []

            base_by_internal = {ik: b for b, _, ik in matched}

            for base_id, mediator, internal_key in matched:
                per_req = saves_by_req.setdefault(base_id, {})
                frame = mediator.info.frame
                for key, value in frame.f_locals.items():
                    if id(value) in Globals.saves:
                        stripped, has_real, has_lazy = strip_lazy(value)
                        # Purely owned by another rank — that rank ships the
                        # real data; contribute nothing (engine merges).
                        if has_lazy and not has_real:
                            continue
                        per_req[key] = stripped
                        if internal_key in finished_internal_keys:
                            removals.append(id(value))

            for internal_key in finished_internal_keys:
                owning_base = base_by_internal.get(internal_key, internal_key)
                for _, ctx in self.trace_contexts.items():
                    if internal_key in ctx["pending_req_ids"]:
                        ctx["pending_req_ids"].discard(internal_key)
                        trace_fully_done = (
                            not ctx["pending_req_ids"]
                            and ctx["received_count"] == ctx["expected_count"]
                        )
                        if trace_fully_done:
                            canonical = ctx["canonical_globals"]
                            per_req = saves_by_req.setdefault(owning_base, {})
                            for name in ctx["saved_names"]:
                                if name in canonical:
                                    value = canonical[name]
                                    if id(value) in Globals.saves:
                                        stripped, has_real, has_lazy = strip_lazy(value)
                                        if has_lazy and not has_real:
                                            continue
                                        per_req[name] = stripped
                                        removals.append(id(value))
                        break

            return saves_by_req, removals

        def cleanup_finished(self, finished_internal_keys: set, removals: list) -> None:
            """Clean up state for finished requests.

            Discards collected IDs from ``Globals.saves``, deletes
            completed trace contexts, and drops mediator entries.
            """
            for _id in removals:
                Globals.saves.discard(_id)

            done_traces = [
                tid
                for tid, ctx in self.trace_contexts.items()
                if (
                    not ctx["pending_req_ids"]
                    and ctx["received_count"] == ctx["expected_count"]
                )
            ]
            for tid in done_traces:
                del self.trace_contexts[tid]

            for internal_key in finished_internal_keys:
                self.mediators.pop(internal_key, None)

    def __init__(self, *args, **kwargs):

        from .. import VLLM

        super().__init__(*args, **kwargs)

        self.nnsight_model: VLLM

        self.nnsight_request_helper = self.NNsightRequestHelper()

    def load_model(self, *args, **kwargs) -> None:

        from .. import VLLM

        super().load_model(*args, **kwargs)

        # vLLM's LlamaForCausalLM gates ``logits_processor`` inside
        # ``if get_pp_group().is_last_rank:`` without an else-branch
        # ``PPMissingLayer()`` stub — unlike Qwen2/GPT2/OPT/Pythia/
        # Bloom/Gemma2, which construct it unconditionally. On a non-last
        # PP rank the attribute is therefore simply absent, so the
        # dumper-side meta model (built without a real PP group, hence
        # ``is_last_rank`` defaults True and the attribute exists) ships a
        # ``Module:model.logits_processor`` persistent id that this rank
        # cannot resolve → ``UnpicklingError`` on the first request.
        # Inserting the stub here (BEFORE the envoy tree is built on the
        # next line) makes the envoy walk register
        # ``Module:model.logits_processor`` symmetrically. The forward
        # returns early on non-last ranks before the logits_processor call
        # site, so the stub is functionally inert at runtime.
        # See vllm/model_executor/models/llama.py:538-553 (vllm 0.19.1).
        if get_pp_group().world_size > 1:
            from vllm.model_executor.models.llama import LlamaForCausalLM
            from vllm.model_executor.models.utils import PPMissingLayer
            if (isinstance(self.model, LlamaForCausalLM)
                    and not hasattr(self.model, "logits_processor")):
                self.model.logits_processor = PPMissingLayer()

        self.nnsight_model = VLLM(self.model)

        self.nnsight_model.tokenizer = cached_tokenizer_from_config(self.model_config)

        self.nnsight_model.interleaver.mediators = []

        self.nnsight_model.interleaver.batcher = VLLMBatcher()

        self.nnsight_model.interleaver.defer_exceptions = True

        # Mount ``Object.save`` in the worker subprocess. The driver
        # process gets its mount via ``Tracer.__init__``/``__setstate__``
        # (the client constructs Tracers, the serve handler unpickles
        # them), but vLLM workers receive only deserialized Mediators —
        # they never touch a Tracer. Without this call, user code's
        # ``tensor.save()`` AttributeErrors inside the worker thread.
        from ....intervention.tracing.globals import _ensure_mounted

        _ensure_mounted()

        # Only wrap when TP > 1: registers hooks that handle
        # gather/split of sharded tensors and CUDA synchronization
        # for TP-parallel modules.  With TP == 1 nothing is sharded
        # so wrapping is pure overhead.
        if get_tp_group().world_size > 1:
            self.nnsight_model.interleaver.batcher.wrap(self.nnsight_model)

        # ----- Pipeline-parallel setup -----
        pp_world_size = get_pp_group().world_size
        self.pp_enabled = pp_world_size > 1
        self.pp_hook_buffer: Dict[Any, Any] = {}
        self.pp_buffer_condition = threading.Condition()

        if self.pp_enabled:
            from ..pp import PPModuleMap
            from ..pp_listener import PPListener
            import torch.distributed as dist

            num_layers = self.model_config.hf_config.num_hidden_layers
            self.pp_module_map = PPModuleMap(num_layers, pp_world_size)

            # Graft children of meta-model PPMissingLayer stubs onto the
            # worker Envoy tree so users can access ``model.layers[5].attn.output``
            # on a stage that doesn't own layer 5. The meta model was built in
            # ``GPUWorker.__init__`` before distributed init (PP=1, TP=1, full
            # architecture). The lazy-tensor short-circuit kicks in on those
            # grafted envoys via :func:`pp_envoy._is_pp_missing`.
            meta_model = getattr(self, "_pp_meta_model", None)
            if meta_model is not None:
                self._graft_pp_missing_envoys(meta_model)
                del self._pp_meta_model

            # Allgather per-module dtype across PP ranks so every rank can
            # size pull recv-buffers and build LazyRemoteTensor placeholders.
            # The module output shape is learned lazily from the first legacy
            # pull of each module (see ``PPListener._cache_module_shapes``)
            # rather than probed up front — a FakeTensorMode forward over the
            # real TP-sharded model collides with vLLM's ``BasevLLMParameter``
            # ``__torch_function__`` on ``aten.t`` and never completes.
            self.pp_module_meta = self._exchange_pp_module_meta()

            # Dedicated gloo group for pull requests — separate from
            # vLLM's own PP groups so the listener thread's recv() doesn't
            # conflict with vLLM's PP communication. ``new_group`` is
            # collective: ALL ranks in the default group must call it the
            # same number of times. With TP > 1 we have multiple PP groups
            # (one per TP slice), and we must call ``new_group`` once per
            # PP-rank list, mirroring vLLM's ``GroupCoordinator.__init__``.
            tp_size = get_tp_group().world_size
            my_pull_group = None
            for tp_offset in range(tp_size):
                pp_ranks_for_tp = [
                    pp_rank * tp_size + tp_offset
                    for pp_rank in range(pp_world_size)
                ]
                g = dist.new_group(ranks=pp_ranks_for_tp, backend="gloo")
                if dist.get_rank() in pp_ranks_for_tp:
                    my_pull_group = g
            self.pp_pull_group = my_pull_group

            local_rank = get_pp_group().rank_in_group

            self.pp_listener = PPListener(
                buffer=self.pp_hook_buffer,
                condition=self.pp_buffer_condition,
                pull_group=self.pp_pull_group,
                local_rank=local_rank,
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                meta_map=self.pp_module_meta,
            )
            self.pp_listener.start()

            # Belt-and-braces shutdown: if the worker is torn down without
            # a graceful ``collect_nnsight`` finish path, atexit fires and
            # signals the listener to break out of its ``dist.recv`` loop.
            # Without this, the daemon listener thread can busy-loop at
            # 100% CPU after the dist context dies — orphaned worker
            # processes that ignore SIGKILL.
            import atexit
            atexit.register(self.pp_listener.stop)

            # Pin PP fields on the interleaver instance so ``pp_eproperty``
            # and ``Mediator.handle_value_event`` can find them without
            # passing them through every call.
            interleaver = self.nnsight_model.interleaver
            interleaver.pp_enabled = True
            interleaver.pp_local_rank = local_rank
            interleaver.pp_module_map = self.pp_module_map
            interleaver.pp_hook_buffer = self.pp_hook_buffer
            interleaver.pp_buffer_condition = self.pp_buffer_condition
            interleaver.pp_module_meta = self.pp_module_meta
            interleaver.pp_listener = self.pp_listener
        else:
            self.pp_module_map = None
            self.pp_module_meta = {}
            self.pp_listener = None

    def _graft_pp_missing_envoys(self, meta_model: torch.nn.Module) -> None:
        """Graft child envoys from meta model onto PPMissing layer envoys.

        :class:`PPMissingLayer` stubs have no children, so the worker's
        Envoy tree is missing sub-module envoys for non-local layers.
        Grafting the meta model's children (full architecture, PP=1)
        onto each PPMissing envoy lets users access e.g.
        ``model.layers[5].attn.output`` even when layer 5 lives on
        another stage.

        The grafted child envoys wrap meta-device modules.
        :func:`pp_envoy._is_pp_missing` detects them as non-local via
        ``pp_module_map`` and returns LazyRemoteTensors on access.
        """
        from ..pp import is_pp_missing
        from ....intervention.envoy import Envoy

        meta_modules = dict(meta_model.named_modules())

        def graft(envoy):
            if is_pp_missing(envoy._module):
                meta_module = meta_modules.get(envoy.path)
                if meta_module is not None:
                    for name, child_module in meta_module.named_children():
                        child_envoy = Envoy(
                            child_module,
                            path=f"{envoy.path}.{name}",
                            rename=envoy._alias.rename if envoy._alias is not None else None,
                            interleaver=envoy._interleaver,
                        )
                        if hasattr(Envoy, name):
                            envoy._handle_overloaded_mount(child_envoy, name)
                        else:
                            object.__setattr__(envoy, name, child_envoy)
            for child_envoy in envoy._children:
                graft(child_envoy)

        graft(self.nnsight_model)

    def _exchange_pp_module_meta(self) -> dict:
        """Allgather per-module dtype across PP ranks.

        Each rank contributes ``{path: {dtype, num_outputs, module_shapes}}``
        for its local (non-PPMissing) modules. ``dtype`` comes from the
        module's own parameters (no forward needed); ``module_shapes``
        starts empty and is filled in lazily on the first legacy pull of
        each module (``PPListener._cache_module_shapes``). The merged map
        is identical on every rank and lets the listener size pull
        recv-buffers and build LazyRemoteTensor placeholders.
        """
        import torch.distributed as dist
        from ..pp import is_pp_missing

        pp_group = get_pp_group()

        local_meta = {}
        for name, module in self.model.named_modules():
            if not is_pp_missing(module):
                param = next(module.parameters(recurse=False), None)
                dtype = param.dtype if param is not None else self.model_config.dtype
                local_meta[name] = {
                    "dtype": dtype,
                    "num_outputs": 1,
                    "module_shapes": [],
                }

        local_bytes = pickle.dumps(local_meta)
        local_tensor = torch.tensor(
            list(local_bytes), dtype=torch.uint8, device="cpu"
        )
        size_tensor = torch.tensor([len(local_bytes)], dtype=torch.int64)

        all_sizes = [
            torch.zeros(1, dtype=torch.int64)
            for _ in range(pp_group.world_size)
        ]
        dist.all_gather(all_sizes, size_tensor, group=pp_group.cpu_group)

        max_size = max(s.item() for s in all_sizes)
        padded = torch.zeros(max_size, dtype=torch.uint8)
        padded[: len(local_bytes)] = local_tensor

        all_padded = [
            torch.zeros(max_size, dtype=torch.uint8)
            for _ in range(pp_group.world_size)
        ]
        dist.all_gather(all_padded, padded, group=pp_group.cpu_group)

        merged = {}
        for buf, size in zip(all_padded, all_sizes):
            rank_meta = pickle.loads(buf[: size.item()].numpy().tobytes())
            merged.update(rank_meta)

        return merged

    def _pp_wait_for_mediators(self):
        """The one PP sync point: hold the forward until every scheduled mediator
        is AHEAD of it.

        Run at the tail of ``_update_states`` (after ``process_batch_groups``
        set this step's mediators, before ``super().execute_model`` fires module
        hooks). The worker is never made to wait for the forward — it runs ahead
        freely; only the forward waits for the worker, so a local one-shot hook
        is always registered before the forward reaches its module (otherwise
        the monotonic iteration tracker would advance past it and the hook would
        never fire — a permanent hang).

        This forward is for iteration ``k = _pp_scheduled_count - 1`` of the
        request. A mediator is "ahead" for it when:

        - it has already moved PAST iteration ``k`` (``mediator.iteration > k``
          — the worker ran ahead, so its iteration-``k`` hooks were registered
          and consumed), or
        - it is ON iteration ``k`` and has reached its local part (parked at a
          local request, ``event_queue.has_value``) or determined it has none
          (``_pp_past_local``), or
        - it has finished (``not alive``).

        Comparing the worker's own ``iteration`` against ``k`` is what makes the
        per-iteration ``_pp_past_local`` / ``has_value`` flags safe to read: a
        stale flag from a previous step is ignored because the worker has
        advanced past ``k``, and a not-yet-started next step is correctly waited
        on (``iteration == k`` but not yet settled). We wait while the worker is
        still in iteration ``k``'s leading-remote phase (or lagging behind),
        which resolves on the producing rank independently of this gate.
        Bounded so a genuine deadlock errors loudly instead of hanging.
        """
        interleaver = self.nnsight_model.interleaver

        def _ahead(m, k):
            if not m.alive:
                return True
            # This worker never reaches iteration k (e.g. a single-shot
            # ``model.trace`` once the engine generates past its one
            # intervention) — don't wait for it.
            if k > getattr(m, "_pp_max_iteration", 0):
                return True
            # ``_pp_worker_iteration`` (not ``iteration``, which a one-shot hook
            # clears to None) reliably tells which iteration the worker is on.
            it = getattr(m, "_pp_worker_iteration", 0)
            if it > k:
                return True
            return it == k and (
                m.event_queue.has_value or getattr(m, "_pp_past_local", False)
            )

        deadline = time.monotonic() + 30.0
        for mediator in list(interleaver.mediators):
            k = getattr(mediator, "_pp_scheduled_count", 0) - 1
            while not _ahead(mediator, k):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"PP readiness gate: mediator {mediator.name} not ahead "
                        f"of forward (worker_iteration="
                        f"{getattr(mediator, '_pp_worker_iteration', 0)}, k={k}) "
                        f"within 30s"
                    )
                time.sleep(0.0001)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        super()._update_states(scheduler_output)

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

        # Use input_batch.req_ids for the actual batch order after
        # condense()/reorder, not the scheduler dict order.
        self.nnsight_request_helper._batch_req_ids = list(self.input_batch.req_ids)
        self.nnsight_request_helper._num_scheduled_tokens = dict(
            scheduler_output.num_scheduled_tokens
        )

        self.nnsight_request_helper.process_batch_groups(
            scheduler_output.num_scheduled_tokens,
            self.input_batch.req_ids,
            self.nnsight_model,
        )

        self.nnsight_model.interleaver.batcher.needs_batching = (
            len(self.nnsight_model.interleaver.mediators) > 1
        )

        # The one PP sync point: hold this forward until every scheduled
        # mediator (which runs ahead on its own) has reached its local part or
        # determined it has none. AFTER process_batch_groups so the gate sees
        # this step's mediators.
        if self.pp_enabled:
            self._pp_wait_for_mediators()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):

        return_value = None
        interleaver = self.nnsight_model.interleaver

        with interleaver:

            # The per-step bidirectional rendezvous (bump the forward counter,
            # wake workers parked at an iteration boundary, then hold the
            # forward until every scheduled mediator has parked) lives in
            # ``_update_states`` — it must run AFTER ``process_batch_groups``
            # has set this step's mediators / token counts, which happens
            # inside ``super().execute_model`` below. Here we just run the
            # forward; one-shot hooks deliver to whichever mediators are parked.
            return_value = super().execute_model(scheduler_output, intermediate_tensors)

            self.nnsight_request_helper.unflatten(self.nnsight_model)

            # Bound GPU memory: after each forward pass, migrate this rank's
            # pp_hook_buffer clones from GPU to CPU. The buffer accumulates
            # one (hidden, residual) clone per (accessed module, iteration)
            # until the request finishes (collect_nnsight clears it), so on
            # GPU it would grow O(modules x tokens) for long generations and
            # OOM. Moving to CPU after the forward keeps GPU resident to a
            # single forward's worth of clones; CPU RAM absorbs the
            # accumulation. Correctness is preserved for every cross-stage
            # pull — the listener already .cpu()s buffer values when serving
            # (pp_listener.py), so a CPU-resident entry serves pulls
            # unchanged (and skips the pull-time D2H). Migration is off the
            # forward's compute critical path (the forward has returned) and
            # held under the buffer condition so a concurrent listener read
            # isn't torn.
            if self.pp_enabled:
                self._migrate_pp_buffer_to_cpu()

        # Safety net: if ``__enter__`` raised or the forward pass was
        # interrupted before ``return_value`` was assigned, ship back a
        # minimal valid ``ModelRunnerOutput`` so vLLM does not segfault.
        #
        # BUT a ``None`` return is also vLLM's *legitimate* deferred-sampling
        # signal: ``super().execute_model`` returns ``None`` on the sampling
        # rank after stashing ``self.execute_model_state``, expecting
        # ``sample_tokens()`` to be called next (which our ``sample_tokens``
        # override consumes). Masking that ``None`` with a synthetic output
        # makes the Ray distributed executor treat it as terminal and skip
        # ``sample_tokens()``, leaving ``execute_model_state`` unconsumed — the
        # next ``execute_model`` then raises "sample_tokens() must be called
        # after execute_model() returns None." (The multiproc executor calls
        # ``sample_tokens`` regardless, so it tolerated the masking; Ray does
        # not.) Only synthesize when there is NO pending deferral, i.e. a
        # genuine error/interrupt where ``execute_model_state`` is unset.
        if return_value is None and self.execute_model_state is None:
            from vllm.v1.outputs import ModelRunnerOutput

            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            return_value = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            )

        return return_value

    def _migrate_pp_buffer_to_cpu(self):
        """Move accumulated pp_hook_buffer entries from GPU to CPU.

        Called once per forward pass (after the forward returns). Recurses
        into the ``(hidden, residual)`` tuples that layer ``.output`` values
        are. Held under ``pp_buffer_condition`` so the listener thread does
        not read a half-migrated dict; a listener that already obtained a
        value keeps its own tensor reference across the dict reassignment, so
        in-flight pulls are unaffected.
        """
        def _to_cpu(v):
            if isinstance(v, torch.Tensor):
                return v.cpu() if v.is_cuda else v
            if isinstance(v, tuple):
                return tuple(_to_cpu(x) for x in v)
            if isinstance(v, list):
                return [_to_cpu(x) for x in v]
            if isinstance(v, dict):
                return {k: _to_cpu(x) for k, x in v.items()}
            return v

        buf = self.pp_hook_buffer
        with self.pp_buffer_condition:
            for k in list(buf.keys()):
                buf[k] = _to_cpu(buf[k])

    def sample_tokens(self, *args, **kwargs):

        interleaver = self.nnsight_model.interleaver

        with interleaver:

            if self.execute_model_state is not None:

                logits = type(self.nnsight_model).logits.provide(
                    self.nnsight_model,
                    self.execute_model_state.logits,
                )

                state = self.execute_model_state

                self.execute_model_state = type(state)(
                    **{**state._asdict(), "logits": logits}
                )

        return super().sample_tokens(*args, **kwargs)

    def _sample(self, *args, **kwargs):

        sampler_output = None
        interleaver = self.nnsight_model.interleaver

        with interleaver:

            sampler_output = super()._sample(*args, **kwargs)

            sampler_output.sampled_token_ids = type(self.nnsight_model).samples.provide(
                self.nnsight_model,
                sampler_output.sampled_token_ids,
            )

        return sampler_output

    def collect_nnsight(
        self,
        req_ids: list[str],
        finished_req_ids: list[str] | None = None,
    ) -> Optional[bytes]:
        """Collect saved values from mediators, optionally finalizing finished requests.

        Called on every streamed output (async) or on finished requests
        (sync). Saves are collected for ALL ``req_ids``. Mediators listed
        in ``finished_req_ids`` are additionally finalized (result handler,
        cancel) and cleaned up. With PP > 1, every PP stage contributes
        its rank's saves; the engine merges across stages.
        """
        # Only TP-rank-0 of each PP stage returns data — TP siblings
        # carry replicated mediator state and would duplicate. PP-non-zero
        # ranks still contribute saves when PP > 1 (engine merges).
        # ``rank_in_group`` is the rank WITHIN the TP group (0..TP-1);
        # ``rank`` is the global rank (would gate everything but global 0).
        if get_tp_group().rank_in_group != 0:
            return None

        if finished_req_ids is None:
            finished_req_ids = []

        helper = self.nnsight_request_helper
        req_id_set = set(req_ids) | set(finished_req_ids)
        finished_req_id_set = set(finished_req_ids)

        matched = helper.match_req_ids(req_id_set)

        # Signal mediators to exit their iteration loops and wait for
        # their worker threads to die. Ensures all in-flight pulls finish
        # before we finalize or clear the buffer.
        if finished_req_ids and self.pp_enabled:
            interleaver = self.nnsight_model.interleaver
            if hasattr(interleaver, "stop_iteration"):
                interleaver.stop_iteration()
            for _, mediator, _ in matched:
                if mediator.worker is not None:
                    mediator.worker.join(timeout=5.0)

        finished_keys = helper.finalize_mediators(
            matched, finished_req_id_set, self.nnsight_model
        )
        saves_by_req, removals = helper.collect_saves(matched, finished_keys)
        helper.cleanup_finished(finished_keys, removals)

        # Scoped clear: drop this request's composite-key entries ONLY.
        # A blanket clear would also wipe concurrent in-flight requests'
        # slices and break their cross-rank pulls.
        if self.pp_enabled and finished_keys and self.pp_listener is not None:
            # Diagnostic (env-gated, inert otherwise): report buffer size
            # before/after the scoped clear so we can distinguish a real
            # cross-request leak (post-clear size grows over time) from
            # benign allocator caching, and measure the intra-request peak
            # (pre-clear size for a long-generation request).
            import os as _os
            if _os.environ.get("NNSIGHT_PP_BUFFER_DEBUG"):
                def _nbytes(v):
                    # Recurse: layer .output is a (hidden, residual) tuple, so
                    # bare-tensor counting undercounts to zero.
                    if isinstance(v, torch.Tensor):
                        return v.element_size() * v.nelement()
                    if isinstance(v, (tuple, list)):
                        return sum(_nbytes(x) for x in v)
                    if isinstance(v, dict):
                        return sum(_nbytes(x) for x in v.values())
                    return 0
                def _devs(d):
                    out = set()
                    def walk(v):
                        if isinstance(v, torch.Tensor):
                            out.add(str(v.device))
                        elif isinstance(v, (tuple, list)):
                            for x in v:
                                walk(x)
                        elif isinstance(v, dict):
                            for x in v.values():
                                walk(x)
                    for v in d.values():
                        walk(v)
                    return ",".join(sorted(out)) or "-"
                buf = self.pp_hook_buffer
                _pre_n = len(buf)
                _pre_b = sum(_nbytes(v) for v in buf.values())
                _pre_dev = _devs(buf)
                _rank = get_pp_group().rank_in_group
                self.pp_listener.clear_buffer(req_ids=finished_keys)
                _post_n = len(buf)
                _post_b = sum(_nbytes(v) for v in buf.values())
                print(
                    f"[PPBUF rank{_rank}] pre n={_pre_n} {_pre_b/1e6:.2f}MB "
                    f"dev=[{_pre_dev}] -> post n={_post_n} {_post_b/1e6:.2f}MB "
                    f"(finished {len(finished_keys)} reqs)",
                    flush=True,
                )
            else:
                self.pp_listener.clear_buffer(req_ids=finished_keys)

        # Collect deferred exceptions. Each mediator has its own — nest the
        # typed envelope inside THAT request's sub-dict under
        # ``__nnsight_exceptions__`` in the ``{base_id: DeferredError}``
        # shape the client's ``surface_server_errors`` understands. The
        # dynamic NNsightException subclass can't be pickled, so the helper
        # ships plain strings (type_name, message, traceback, is_control_flow).
        from ....intervention.errors import capture_deferred

        for base_id, mediator, _internal_key in matched:
            entry = capture_deferred(mediator, req_id=base_id)
            if entry is not None:
                per_req = saves_by_req.setdefault(base_id, {})
                per_req["__nnsight_exceptions__"] = {base_id: entry}
                mediator.deferred_exception = None
                mediator._deferred_type_name = None
                mediator._deferred_traceback = None
                mediator._deferred_is_control_flow = False

        torch.cuda.synchronize()
        return _ZSTD_COMPRESSOR.compress(pickle.dumps(saves_by_req))
