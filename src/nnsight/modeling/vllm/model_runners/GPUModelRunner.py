import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

                mediator = load(
                    extra_args["nnsight_mediator"],
                    model._remoteable_persistent_objects(),
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

                mediator.idx = len(model.interleaver.mediators)
                model.interleaver.mediators.append(mediator)
                mediator.start(model.interleaver)

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
                    # Non-NNsight request or already-finished mediator —
                    # still occupies a row in the logits tensor.
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
                model.interleaver.batcher.last_batch_group = mediators[-1].batch_group
            else:
                model.interleaver.batcher.last_batch_group = None

            model.interleaver.mediators = mediators

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
                    model.interleaver.mediators = [mediator]
                    mediator.batch_group = None
                    with model.interleaver:
                        model.interleaver.handle("result", [base_id])
                        mediator.cancel()
                        model.interleaver.handle()
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

        self.nnsight_model.interleaver.mediators = []

        self.nnsight_model.interleaver.batcher = VLLMBatcher()

        # Only wrap when TP > 1: registers hooks that handle
        # gather/split of sharded tensors and CUDA synchronization
        # for TP-parallel modules.  With TP == 1 nothing is sharded
        # so wrapping is pure overhead.

        if get_tp_group().world_size > 1:
            self.nnsight_model.interleaver.batcher.wrap(self.nnsight_model)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        super()._update_states(scheduler_output)

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

        # Use input_batch.req_ids for the actual batch order after
        # condense()/reorder, not the scheduler dict order.
        # Store these for unflatten() which needs the same ordering.
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

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):

        Globals.enter()
        with self.nnsight_model.interleaver:

            return_value = super().execute_model(scheduler_output, intermediate_tensors)

            self.nnsight_request_helper.unflatten(self.nnsight_model)

        Globals.exit()

        return return_value

    def sample_tokens(self, *args, **kwargs):

        Globals.enter()

        with self.nnsight_model.interleaver:

            # Provide logits from execute_model state before sampling.
            if self.execute_model_state is not None:

                logits = type(self.nnsight_model).logits.provide(
                    self.nnsight_model,
                    self.execute_model_state.logits,
                )

                state = self.execute_model_state

                self.execute_model_state = type(state)(
                    **{**state._asdict(), "logits": logits}
                )

        Globals.exit()

        return super().sample_tokens(*args, **kwargs)

    def _sample(self, *args, **kwargs):

        Globals.enter()

        with self.nnsight_model.interleaver:

            sampler_output = super()._sample(*args, **kwargs)

            sampler_output.sampled_token_ids = type(self.nnsight_model).samples.provide(
                self.nnsight_model,
                sampler_output.sampled_token_ids,
            )

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
        if get_pp_group().rank != 0:
            return None

        if finished_req_ids is None:
            finished_req_ids = []

        helper = self.nnsight_request_helper
        req_id_set = set(req_ids) | set(finished_req_ids)
        finished_req_id_set = set(finished_req_ids)

        matched = helper.match_req_ids(req_id_set)
        finished_keys = helper.finalize_mediators(
            matched, finished_req_id_set, self.nnsight_model
        )
        saves, removals = helper.collect_saves(matched, finished_keys)
        helper.cleanup_finished(finished_keys, removals)

        return pickle.dumps(saves)
