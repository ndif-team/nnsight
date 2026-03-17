import base64
import logging
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class NNsightRequestHelper:
    """Manages the lifecycle of NNsight mediators for SGLang requests.

    Handles deserialization, batch group mapping, save collection, and cleanup.
    Parallels vLLM's ``NNsightGPUModelRunner.NNsightRequestHelper``.
    """

    def __init__(self):
        self.req_id_to_mediator: Dict[str, Any] = {}
        self.trace_contexts: Dict[str, dict] = {}
        self.pending_saves: Dict[str, bytes] = {}

    def register_request(
        self, rid: str, nnsight_data_str: str, nnsight_model
    ) -> None:
        """Deserialize a mediator from the custom_logit_processor string."""
        from nnsight.intervention.serialization import load
        from nnsight.intervention.tracing.globals import Globals

        # Decode the nnsight data
        raw = nnsight_data_str[len("nnsight:"):]
        data = pickle.loads(base64.b64decode(raw))

        # Build persistent objects dict for deserialization.
        # This maps persistent IDs (like "Module:model", "Tokenizer")
        # to the actual objects in the subprocess.
        persistent_objects = nnsight_model._remoteable_persistent_objects()
        if hasattr(nnsight_model, 'tokenizer') and nnsight_model.tokenizer is not None:
            persistent_objects["Tokenizer"] = nnsight_model.tokenizer
        print(f"[NNsight] Subprocess persistent object keys (first 10): {sorted(persistent_objects.keys())[:10]}", flush=True)
        mediator = load(data["nnsight_mediator"], persistent_objects)

        trace_id = data["nnsight_trace_id"]
        saved_names = data.get("nnsight_saved_names", [])

        if trace_id not in self.trace_contexts:
            canonical_globals = mediator.intervention.__globals__
            for name in saved_names:
                if name in canonical_globals:
                    Globals.saves.add(id(canonical_globals[name]))

            self.trace_contexts[trace_id] = {
                "saved_names": saved_names,
                "canonical_globals": canonical_globals,
                "expected_count": data.get("nnsight_expected_count", 1),
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

        nnsight_model._interleaver.mediators.append(mediator)
        mediator.start(nnsight_model._interleaver)

        self.req_id_to_mediator[rid] = mediator
        ctx["pending_req_ids"].add(rid)
        ctx["received_count"] += 1

    def setup_batch_groups(
        self, reqs, nnsight_model
    ) -> None:
        """Map batch positions to mediators based on request ordering.

        In decode mode each request contributes 1 token to the batch.
        In extend mode each request contributes len(fill_ids) tokens.
        """
        batch_start = 0
        mediators = []

        for req in reqs:
            rid = req.rid
            # Determine how many tokens this request contributes
            num_tokens = 1  # decode mode default

            mediator = self.req_id_to_mediator.get(rid)
            if mediator is not None:
                mediators.append(mediator)
                mediator.batch_group = [batch_start, num_tokens]

            batch_start += num_tokens

        if mediators:
            nnsight_model._interleaver.batcher.last_batch_group = mediators[-1].batch_group
        else:
            nnsight_model._interleaver.batcher.last_batch_group = None

        nnsight_model._interleaver.mediators = mediators
        nnsight_model._interleaver.batcher.needs_batching = len(mediators) > 1

    def finalize_request(
        self, rid: str, nnsight_model
    ) -> None:
        """Finalize a finished request: run result handler, cancel mediator, collect saves."""
        from nnsight.intervention.tracing.globals import Globals

        mediator = self.req_id_to_mediator.get(rid)
        if mediator is None:
            return

        Globals.enter()
        if mediator.alive:
            nnsight_model._interleaver.mediators = [mediator]
            mediator.batch_group = None
            with nnsight_model._interleaver:
                nnsight_model._interleaver.handle("result", [rid])
                mediator.cancel()
                nnsight_model._interleaver.handle()
        Globals.exit()

    def collect_and_cleanup(
        self, rids: List[str], nnsight_model
    ) -> Optional[bytes]:
        """Collect saved values for the given request IDs and clean up."""
        from nnsight.intervention.tracing.globals import Globals

        saves = {}
        removals: Set[int] = set()

        for rid in rids:
            mediator = self.req_id_to_mediator.get(rid)
            if mediator is None:
                continue

            # Finalize if not already done
            self.finalize_request(rid, nnsight_model)

            # Collect from frame locals
            frame = mediator.info.frame
            for key, value in frame.f_locals.items():
                if id(value) in Globals.saves:
                    saves[key] = value
                    removals.add(id(value))

        # Collect trace-shared saves
        finished_rids = set(rids)
        for rid in finished_rids:
            for tid, ctx in self.trace_contexts.items():
                if rid in ctx["pending_req_ids"]:
                    ctx["pending_req_ids"].discard(rid)
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

        # Cleanup
        for _id in removals:
            Globals.saves.discard(_id)

        done_traces = [
            tid for tid, ctx in self.trace_contexts.items()
            if (not ctx["pending_req_ids"]
                and ctx["received_count"] == ctx["expected_count"])
        ]
        for tid in done_traces:
            del self.trace_contexts[tid]

        for rid in rids:
            self.req_id_to_mediator.pop(rid, None)

        if saves:
            return pickle.dumps(saves)
        return None


class NNsightModelRunner(ModelRunner):
    """ModelRunner subclass that wraps forward passes with NNsight's interleaver.

    After ``load_model()``, the underlying PyTorch model is wrapped with an
    ``NNsight`` Envoy. Forward passes (``forward_decode`` / ``forward_extend``)
    are wrapped with the interleaver context so hooks fire on all modules.
    """

    def __init__(self, *args, **kwargs):
        print(f"[NNsight] NNsightModelRunner.__init__ called", flush=True)
        self.nnsight_model = None
        self.nnsight_request_helper = NNsightRequestHelper()
        super().__init__(*args, **kwargs)
        print(f"[NNsight] NNsightModelRunner.__init__ done, nnsight_model={self.nnsight_model is not None}", flush=True)

    def load_model(self):
        super().load_model()

        from ..sglang import SGLang
        from nnsight.util import WrapperModule
        from .batching import SGLangBatcher
        from transformers import AutoTokenizer

        logger.info(f"NNsight wrapping model: {type(self.model).__name__}")

        # Create a full SGLang wrapper (RemoteableMixin) so that
        # _remoteable_persistent_objects() works for mediator deserialization.
        # Passing an nn.Module directly skips _load/_load_meta.
        self.nnsight_model = SGLang(self.model)

        self.nnsight_model._interleaver.mediators = []
        self.nnsight_model._interleaver.batcher = SGLangBatcher()

        # Add WrapperModules for logits and samples access
        self.nnsight_model.logits = WrapperModule()
        self.nnsight_model.samples = WrapperModule()

        # Load tokenizer in subprocess for mediator deserialization
        try:
            self.nnsight_model.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_path,
                trust_remote_code=getattr(self.server_args, 'trust_remote_code', True),
            )
        except Exception as e:
            logger.warning(f"NNsight: Failed to load tokenizer: {e}")
            self.nnsight_model.tokenizer = None

        logger.info("NNsight model runner initialized with Envoy wrapping")

    def forward_decode(self, forward_batch, **kwargs):
        if not self.nnsight_model or not self.nnsight_model._interleaver.mediators:
            return super().forward_decode(forward_batch, **kwargs)

        from nnsight.intervention.tracing.globals import Globals

        # Set up batch groups for decode (each request = 1 token)
        if hasattr(forward_batch, 'reqs') and forward_batch.reqs:
            self.nnsight_request_helper.setup_batch_groups(
                forward_batch.reqs, self.nnsight_model
            )

        Globals.enter()
        try:
            with self.nnsight_model._interleaver:
                result = super().forward_decode(forward_batch, **kwargs)

                # Fire logits hook
                if hasattr(result, 'next_token_logits') and result.next_token_logits is not None:
                    result.next_token_logits = self.nnsight_model.logits(
                        result.next_token_logits, hook=True
                    )
        finally:
            Globals.exit()

        return result

    def forward_extend(self, forward_batch, **kwargs):
        if not self.nnsight_model or not self.nnsight_model._interleaver.mediators:
            return super().forward_extend(forward_batch, **kwargs)

        from nnsight.intervention.tracing.globals import Globals

        # Set up batch groups for extend
        if hasattr(forward_batch, 'reqs') and forward_batch.reqs:
            self.nnsight_request_helper.setup_batch_groups(
                forward_batch.reqs, self.nnsight_model
            )

        Globals.enter()
        try:
            with self.nnsight_model._interleaver:
                result = super().forward_extend(forward_batch, **kwargs)

                # forward_extend returns (logits_output, can_run_graph)
                logits_output = result
                if isinstance(result, tuple):
                    logits_output = result[0]

                if hasattr(logits_output, 'next_token_logits') and logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = self.nnsight_model.logits(
                        logits_output.next_token_logits, hook=True
                    )
        finally:
            Globals.exit()

        return result
