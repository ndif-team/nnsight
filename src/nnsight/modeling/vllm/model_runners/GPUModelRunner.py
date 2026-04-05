import pickle
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.distributed.parallel_state import get_tp_group
from nnsight.intervention.tracing.globals import Globals

from ....intervention.serialization import load
from ..batching import VLLMBatcher

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
    """

    class NNsightRequestHelper:
        """
        Helper class for batching requests in the GPUModelRunner.

        Attributes:
            ids_to_batch_group (Dict[str, int]): Dictionary mapping request IDs to their assigned batch group indices.
            interleaver_to_ids (Dict[Interleaver, Set[str]]): Dictionary mapping interleavers to sets of request IDs.
            flat_batch_groups (Dict[Interleaver, List[Tuple[int, int]]]): Dictionary mapping interleavers to their flattened batch groups.

        Methods:
            process_new_reqs(new_reqs: List[NewRequestData]) -> None: Process new requests and compute the flat batch groups.
            process_finished_req(req_id: str, interleaver: Interleaver) -> None: Process a finished request,
                by updating batch groups and cleaning up mappings.
        """

        def __init__(self):

            self.req_id_to_batch_group_idx: Dict[str, int] = {}
            self.mediators: Dict[str, Any] = {}  # req_id -> Mediator
            self.trace_contexts: Dict[str, dict] = {}  # trace_id -> context

        def _pp_aware_load(self, data: bytes, model: VLLM):
            """Deserialize a mediator with PP-aware persistent ID resolution.

            When PP is enabled, the serialized mediator references module paths
            from the full meta model (e.g. ``model.transformer.h.6.ln_1``).
            On PP workers, layers on other stages are ``PPMissingLayer`` stubs
            with no children.  This method falls back to the nearest ancestor
            ``PPMissingLayer`` stub when a child path cannot be resolved.
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
                    # PP fallback: walk up the module path to find the
                    # nearest ancestor that exists (a PPMissingLayer stub).
                    if isinstance(pid, str) and pid.startswith("Module:"):
                        path = pid[len("Module:"):]
                        parts = path.split(".")
                        for i in range(len(parts) - 1, 0, -1):
                            ancestor_pid = "Module:" + ".".join(parts[:i])
                            if ancestor_pid in self.persistent_objects:
                                ancestor = self.persistent_objects[ancestor_pid]
                                if is_pp_missing(ancestor):
                                    return ancestor
                    raise __import__("pickle").UnpicklingError(
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

            Args:
                new_reqs (List[NewRequestData]): List of new request data objects to process.
            """

            for new_req in new_reqs:

                extra_args = getattr(new_req.sampling_params, "extra_args", None)
                if not extra_args:
                    continue

                trace_id = extra_args.get("nnsight_trace_id")
                if trace_id is None:
                    # Non-NNsight request, skip
                    continue

                mediator = self._pp_aware_load(
                    extra_args["nnsight_mediator"],
                    model,
                )

                saved_names = extra_args.get("nnsight_saved_names", [])

                # First mediator for this trace: create context and register
                # its __globals__ as canonical for shared variable grafting.
                if trace_id not in self.trace_contexts:
                    canonical_globals = mediator.intervention.__globals__

                    # Register saved vars in worker-side Globals.saves
                    # (.save() was called on the client with a different id).
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
                    # Subsequent mediator: graft saved vars from canonical
                    # globals so all mediators share the same Python objects.
                    ctx = self.trace_contexts[trace_id]
                    canonical = ctx["canonical_globals"]
                    med_globals = mediator.intervention.__globals__
                    for name in saved_names:
                        if name in canonical:
                            med_globals[name] = canonical[name]

                ctx = self.trace_contexts[trace_id]

                # Reset the iteration gate for the new request so
                # mediators are not blocked by a previous stop signal.
                model._interleaver._generation_done = False

                model._interleaver.mediators.append(mediator)
                mediator.start(model._interleaver)

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
            mediator_set = {id(m) for m in model._interleaver.mediators}

            for req_id in self._batch_req_ids:
                if self._num_scheduled_tokens.get(req_id) is None:
                    continue

                mediator = self.mediators.get(req_id)

                if mediator is None or id(mediator) not in mediator_set:
                    # Non-NNsight request or already-finished mediator —
                    # still occupies a row in the logits tensor.
                    batch_start += 1
                    continue

                mediator.batch_group = [batch_start, 1]
                batch_start += 1
                model._interleaver.batcher.last_batch_group = mediator.batch_group

        def process_batch_groups(
            self,
            num_tokens_scheduled: Dict[str, int],
            batch_req_ids: List[str],
            model: VLLM,
        ) -> None:

            batch_start = 0

            mediators = []

            # Iterate in input_batch order (batch_req_ids) rather than
            # scheduler dict order, because input_batch.condense() and
            # _may_reorder_batch() can reorder requests after the scheduler
            # builds num_scheduled_tokens.  The model's tensors (including
            # sampled_token_ids) follow input_batch order.
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

                batch_start += num_tokens

            if mediators:
                model._interleaver.batcher.last_batch_group = mediators[-1].batch_group
            else:
                model._interleaver.batcher.last_batch_group = None

            model._interleaver.mediators = mediators

        def match_req_ids(self, req_id_set: set) -> List[tuple]:
            """Match engine-reported request IDs to stored mediators.

            vLLM appends a hash suffix to request IDs (e.g. ``"0-abc123"``
            or ``"uuid-abc123"``).  This method strips the suffix with
            ``rsplit`` and falls back to an exact match.

            Returns:
                List of ``(base_id, mediator, internal_key)`` tuples.
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
            """Run result handler and cancel finished mediators.

            Returns:
                Set of internal keys for mediators that were finalized.
            """
            finished_internal_keys = set()
            for base_id, mediator, internal_key in matched:
                if base_id not in finished_req_id_set:
                    continue

                finished_internal_keys.add(internal_key)

                Globals.enter()
                if mediator.alive:
                    model._interleaver.mediators = [mediator]
                    mediator.batch_group = None
                    with model._interleaver:
                        model._interleaver.handle("result", [base_id])
                        mediator.cancel()
                        model._interleaver.handle()
                Globals.exit()

            return finished_internal_keys

        def collect_saves(self, matched, finished_internal_keys: set) -> tuple:
            """Collect saved values from mediator frames.

            Gathers per-invoke saves from frame locals and trace-shared
            saves from canonical globals (only when a trace is fully done).

            Returns:
                ``(saves, removals)`` — the saves dict and set of
                ``id()`` values to discard from ``Globals.saves``.
            """
            saves = {}
            removals = set()

            for base_id, mediator, internal_key in matched:
                frame = mediator.info.frame
                for key, value in frame.f_locals.items():
                    if id(value) in Globals.saves:
                        saves[key] = value
                        if internal_key in finished_internal_keys:
                            removals.add(id(value))

            # Trace-shared saves: collect when ALL mediators for a trace
            # have been received AND completed.
            for internal_key in finished_internal_keys:
                for _, ctx in self.trace_contexts.items():
                    if internal_key in ctx["pending_req_ids"]:
                        ctx["pending_req_ids"].discard(internal_key)
                        trace_fully_done = (
                            not ctx["pending_req_ids"]
                            and ctx["received_count"] == ctx["expected_count"]
                        )
                        if trace_fully_done:
                            canonical = ctx["canonical_globals"]
                            for name in ctx["saved_names"]:
                                if name in canonical:
                                    value = canonical[name]
                                    if id(value) in Globals.saves:
                                        saves[name] = value
                                        removals.add(id(value))
                        break

            return saves, removals

        def cleanup_finished(self, finished_internal_keys: set, removals: set) -> None:
            """Clean up state for finished requests.

            Removes entries from ``Globals.saves``, deletes completed
            trace contexts, and drops mediator entries.
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

        self.nnsight_model = VLLM(self.model)

        self.nnsight_model.tokenizer = cached_tokenizer_from_config(self.model_config)

        self.nnsight_model._interleaver.mediators = []

        self.nnsight_model._interleaver.batcher = VLLMBatcher()

        # Always call wrap() to register compat transforms (module
        # detection for HF-compatibility layer).  TP gather/split
        # hooks are gated on world_size > 1 inside wrap() itself.
        self.nnsight_model._interleaver.batcher.wrap(self.nnsight_model)

        # --- PP support: buffer, condition, module map ---
        pp_world_size = get_pp_group().world_size
        self.pp_enabled = pp_world_size > 1
        self.pp_hook_buffer: dict[str, Any] = {}
        self.pp_buffer_condition = threading.Condition()

        if self.pp_enabled:
            from ..pp import PPModuleMap

            num_layers = self.model_config.hf_config.num_hidden_layers
            self.pp_module_map = PPModuleMap(num_layers, pp_world_size)

            # Graft meta model's children onto PPMissing envoys so users
            # can access sub-modules (e.g., model.layers[5].attn.output).
            # The meta model was created in GPUWorker.__init__ before
            # distributed init (PP=1, TP=1, full architecture).
            meta_model = getattr(self, '_pp_meta_model', None)
            if meta_model is not None:
                self._graft_pp_missing_envoys(meta_model)
                del self._pp_meta_model

            # Exchange local module metadata across PP ranks to build
            # a complete {path: (dtype, static_shape)} map. Each rank
            # contributes its local (non-PPMissing) modules' shapes,
            # which reflect the real TP/EP sharding.
            self.pp_module_meta = self._exchange_pp_module_meta()
        else:
            self.pp_module_map = None
            self.pp_module_meta = {}

        # --- PP listener with cross-rank pull support ---
        if self.pp_enabled:
            from ..pp_listener import PPListener
            import torch.distributed as dist

            pp_group = get_pp_group()
            local_rank = pp_group.rank_in_group

            # Dedicated gloo group for pull requests — separate from
            # vLLM's PP groups so the listener thread's recv() doesn't
            # conflict with vLLM's PP communication.  Tags separate
            # request vs response traffic within the same group.
            #
            # new_group is collective: ALL ranks in the default group
            # must call it the same number of times. With TP>1, there
            # are multiple PP groups (one per TP slice). We must call
            # new_group for ALL PP group rank lists, matching vLLM's
            # pattern in GroupCoordinator.__init__.
            my_pull_group = None
            tp_size = get_tp_group().world_size
            world_size = dist.get_world_size()
            for tp_offset in range(tp_size):
                pp_ranks_for_tp = [
                    pp_rank * tp_size + tp_offset
                    for pp_rank in range(pp_world_size)
                ]
                g = dist.new_group(ranks=pp_ranks_for_tp, backend="gloo")
                if dist.get_rank() in pp_ranks_for_tp:
                    my_pull_group = g
            self.pp_pull_group = my_pull_group

            self.pp_listener = PPListener(
                buffer=self.pp_hook_buffer,
                condition=self.pp_buffer_condition,
                pull_group=self.pp_pull_group,
                local_rank=local_rank,
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                dtype_map=self.pp_module_meta,
            )
            self.pp_listener.start()
        else:
            self.pp_listener = None

    def _graft_pp_missing_envoys(self, meta_model: torch.nn.Module) -> None:
        """Graft child Envoys from meta model onto PPMissing layer envoys.

        PPMissingLayer stubs have no children, so the worker's Envoy tree
        is missing sub-module envoys for non-local layers. This grafts
        the meta model's children (full architecture, PP=1) onto each
        PPMissing envoy so users can access e.g. model.layers[5].attn.output.

        The grafted child envoys wrap meta-device modules. _is_pp_missing
        detects them as non-local via pp_module_map and returns
        LazyRemoteTensors on .output access.
        """
        from ..pp import is_pp_missing
        from ...intervention.envoy import Envoy

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
        """Exchange local module metadata across PP ranks.

        Each rank collects {path: (dtype, static_shape)} for its local
        (non-PPMissing) modules — shapes reflect real TP/EP sharding.
        An allgather across PP ranks merges into a complete map.

        Returns:
            dict mapping module_path -> dtype (torch.dtype).
            Shape info is encoded separately per module for the pull
            protocol.
        """
        import pickle
        import torch.distributed as dist
        from ..pp import is_pp_missing

        pp_group = get_pp_group()

        # Collect local module metadata
        local_meta = {}
        for name, module in self.model.named_modules():
            if not is_pp_missing(module):
                # Use first parameter's dtype, fall back to model config
                param = next(module.parameters(recurse=False), None)
                dtype = param.dtype if param is not None else self.model_config.dtype
                local_meta[name] = dtype

        # Serialize and allgather across PP ranks
        local_bytes = pickle.dumps(local_meta)
        local_tensor = torch.tensor(
            list(local_bytes), dtype=torch.uint8, device="cpu"
        )
        size_tensor = torch.tensor([len(local_bytes)], dtype=torch.int64)

        # Gather sizes first
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

        # Merge all ranks' metadata
        merged = {}
        for i, (buf, size) in enumerate(zip(all_padded, all_sizes)):
            rank_meta = pickle.loads(buf[: size.item()].numpy().tobytes())
            merged.update(rank_meta)

        return merged

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        super()._update_states(scheduler_output)

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

        # Use input_batch.req_ids for the actual batch order after
        # condense()/reorder, not the scheduler dict order.
        # Store these for unflatten() which needs the same ordering.
        self.nnsight_request_helper._batch_req_ids = list(self.input_batch.req_ids)
        self.nnsight_request_helper._num_scheduled_tokens = dict(scheduler_output.num_scheduled_tokens)

        self.nnsight_request_helper.process_batch_groups(
            scheduler_output.num_scheduled_tokens,
            self.input_batch.req_ids,
            self.nnsight_model,
        )

        self.nnsight_model._interleaver.batcher.needs_batching = (
            len(self.nnsight_model._interleaver.mediators) > 1
        )

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):

        # PP state must be set BEFORE entering the interleaver, because
        # __enter__ starts mediator threads that check _is_pp_missing.
        if self.pp_enabled:
            interleaver = self.nnsight_model._interleaver
            interleaver.pp_enabled = True
            interleaver.pp_local_rank = get_pp_group().rank_in_group
            interleaver.pp_module_map = self.pp_module_map
            interleaver.pp_hook_buffer = self.pp_hook_buffer
            interleaver.pp_buffer_condition = self.pp_buffer_condition
            interleaver.pp_module_meta = self.pp_module_meta
            if getattr(self, 'pp_listener', None) is not None:
                interleaver.pp_listener = self.pp_listener

        Globals.enter()

        return_value = None
        interleaver = self.nnsight_model._interleaver
        interleaver._defer_exceptions = True

        with interleaver:

            # Wait until all mediators are parked at a local module
            # before firing forward pass hooks.  The iteration gate
            # (in IteratorTracer) prevents mediators from starting new
            # iterations, but within an iteration they still need to
            # process PP-missing accesses and reach a local module.
            if getattr(self, 'pp_enabled', False) and self.nnsight_model._interleaver.mediators:
                self._pp_wait_for_mediator_readiness()

            return_value = super().execute_model(scheduler_output, intermediate_tensors)

            self.nnsight_request_helper.unflatten(self.nnsight_model)

            if self.execute_model_state is not None:

                logits = self.nnsight_model.logits(
                    self.execute_model_state.logits, hook=True
                )

                state = self.execute_model_state

                self.execute_model_state = type(state)(
                    **{**state._asdict(), "logits": logits}
                )

        interleaver._defer_exceptions = False

        # Safety net: if __enter__ failed or forward was interrupted before
        # return_value could be assigned, build a minimal valid output.
        if return_value is None:
            from vllm.v1.outputs import ModelRunnerOutput
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            return_value = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            )

        Globals.exit()

        return return_value

    def _pp_wait_for_mediator_readiness(self):
        """Wait until all mediators have a pending event for a local module.

        Mediators run freely for PPMissing accesses (Envoy short-circuit
        returns LazyRemoteTensor without posting to event_queue). Before
        firing forward-pass hooks, we must ensure each mediator has
        finished all PPMissing processing and is blocked waiting for a
        local module (i.e., has an event in its event_queue). This
        prevents hooks from firing with no consumer waiting.
        """
        for mediator in self.nnsight_model._interleaver.mediators:
            while mediator.alive and not mediator.event_queue.has_value:
                time.sleep(0.0001)

    def _sample(self, *args, **kwargs):

        Globals.enter()

        sampler_output = None
        interleaver = self.nnsight_model._interleaver
        interleaver._defer_exceptions = True

        with interleaver:

            sampler_output = super()._sample(*args, **kwargs)

            sampler_output.sampled_token_ids = self.model.samples(
                sampler_output.sampled_token_ids, hook=True
            )

        interleaver._defer_exceptions = False

        Globals.exit()

        return sampler_output

    def collect_nnsight(
        self,
        req_ids: list[str],
        finished_req_ids: list[str] | None = None,
    ) -> Optional[bytes]:
        """Collect saved values from mediators, optionally finalizing finished requests.

        Called on every streamed output (async) or on finished requests (sync).
        Saves are collected for ALL ``req_ids``.  Mediators listed in
        ``finished_req_ids`` are additionally finalized (result handler,
        cancel) and cleaned up.

        Args:
            req_ids: Request IDs to collect current saves from.
            finished_req_ids: Subset of request IDs that are finished and
                should be finalized and cleaned up.  ``None`` means no
                requests are finished.
        """
        if finished_req_ids is None:
            finished_req_ids = []

        helper = self.nnsight_request_helper
        req_id_set = set(req_ids) | set(finished_req_ids)
        finished_req_id_set = set(finished_req_ids)

        matched = helper.match_req_ids(req_id_set)

        # Signal mediators to exit their iteration loops and wait for
        # threads to die.  This ensures all in-flight pulls complete
        # before we finalize or clear the buffer.
        if finished_req_ids:
            interleaver = self.nnsight_model._interleaver
            interleaver.stop_iteration()
            for _, mediator, _ in matched:
                if mediator.worker is not None:
                    mediator.worker.join(timeout=5.0)

        finished_keys = helper.finalize_mediators(
            matched, finished_req_id_set, self.nnsight_model
        )
        saves, removals = helper.collect_saves(matched, finished_keys)
        helper.cleanup_finished(finished_keys, removals)

        # Clear request-scoped buffer entries on completion.
        if getattr(self, 'pp_enabled', False) and finished_keys:
            with self.pp_buffer_condition:
                self.pp_hook_buffer.clear()

        # Collect deferred exceptions from mediators, keyed by request ID.
        # Each invoke has its own mediator and its own exception.
        # The wrapped exception can't be pickled (dynamic class), so send type + message.
        exceptions = {}
        for base_id, mediator, internal_key in matched:
            if mediator._deferred_exception is not None:
                exc = mediator._deferred_exception
                exceptions[base_id] = {
                    "type": type(exc).__bases__[0].__name__ if hasattr(type(exc), "__bases__") else type(exc).__name__,
                    "message": str(exc),
                }
                mediator._deferred_exception = None

        if exceptions:
            saves["__nnsight_exceptions__"] = exceptions

        return pickle.dumps(saves)

    def test_pp_buffer_put(self, entries: dict) -> bytes:
        """Put tensors into this rank's buffer.

        entries: {key: (shape, dtype_str, seed)}
        Generates deterministic data on CPU, then moves to GPU.
        Returns checksums for verification.
        """
        pp_rank = get_pp_group().rank_in_group
        checksums = {}
        for key, (shape, dtype_str, seed) in entries.items():
            dtype = getattr(torch, dtype_str)
            g = torch.Generator().manual_seed(seed)
            tensor = torch.randn(shape, dtype=torch.float32, generator=g).to(dtype).cuda()
            checksums[key] = float(tensor.float().sum().item())
            with self.pp_buffer_condition:
                self.pp_hook_buffer[key] = tensor
                self.pp_buffer_condition.notify_all()
        return pickle.dumps({"rank": pp_rank, "keys": list(entries.keys()), "checksums": checksums})

    def test_pp_pull(self, source_rank: int, key: str, shape: list, dtype_str: str, seed: int) -> bytes:
        """Pull a tensor from source_rank and verify via checksum.
        No-op if this rank IS the source rank.
        """
        pp_rank = get_pp_group().rank_in_group
        if pp_rank == source_rank:
            return pickle.dumps({"rank": pp_rank, "role": "server"})
        result = self.pp_listener.pull_from_remote(source_rank, key)
        # Recompute expected checksum the same way the server did
        dtype = getattr(torch, dtype_str)
        g = torch.Generator().manual_seed(seed)
        expected = torch.randn(shape, dtype=torch.float32, generator=g).to(dtype)
        expected_sum = float(expected.float().sum().item())
        result_sum = float(result.float().cpu().sum().item())
        match = abs(result_sum - expected_sum) < 1e-2
        return pickle.dumps({
            "rank": pp_rank, "match": match,
            "shape": list(result.shape),
            "device": str(result.device),
            "dtype": str(result.dtype),
        })

    def test_pp_buffer_clear(self) -> bytes:
        """Clear the buffer."""
        self.pp_hook_buffer.clear()
        return pickle.dumps({"rank": get_pp_group().rank_in_group})

    def test_pp_profile_pull(self, num_pulls: int, shape: list, dtype_str: str, direction: str) -> bytes:
        """Profile pull latency. Rank-aware: one rank serves, the other pulls.

        Args:
            num_pulls: number of sequential pulls to perform
            shape: tensor shape
            dtype_str: e.g. "bfloat16"
            direction: "0to1" (rank 0 serves, rank 1 pulls) or "1to0"
        """
        pp_rank = get_pp_group().rank_in_group
        dtype = getattr(torch, dtype_str)
        numel = 1
        for s in shape:
            numel *= s

        serve_rank = int(direction[0])
        pull_rank = int(direction[-1])

        if pp_rank == serve_rank:
            # Populate buffer with all keys upfront
            for i in range(num_pulls):
                key = f"prof_{i}"
                tensor = torch.randn(shape, dtype=dtype, device="cuda")
                with self.pp_buffer_condition:
                    self.pp_hook_buffer[key] = tensor
                    self.pp_buffer_condition.notify_all()
            # Don't clear here — the listener thread may still be
            # serving the puller. The caller should clear separately
            # via test_pp_buffer_clear after the RPC completes.
            return pickle.dumps({"rank": pp_rank, "role": "server"})

        elif pp_rank == pull_rank:
            torch.cuda.synchronize()
            times = []
            for i in range(num_pulls):
                key = f"prof_{i}"
                t0 = time.perf_counter()
                result = self.pp_listener.pull_from_remote(serve_rank, key)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
            total = sum(times)
            elem_bytes = result.element_size()
            tensor_bytes = numel * elem_bytes
            return pickle.dumps({
                "rank": pp_rank,
                "role": "puller",
                "num_pulls": num_pulls,
                "shape": shape,
                "dtype": dtype_str,
                "tensor_bytes": tensor_bytes,
                "times_ms": [t * 1000 for t in times],
                "total_ms": total * 1000,
                "mean_ms": (total / num_pulls) * 1000,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
            })
        else:
            return pickle.dumps({"rank": pp_rank, "role": "bystander"})
