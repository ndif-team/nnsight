import base64
import logging
import pickle
from typing import Optional

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput, TokenizedGenerateReqInput

logger = logging.getLogger(__name__)


class NNsightScheduler(Scheduler):
    """Scheduler subclass that creates NNsightModelRunner and handles nnsight RPC.

    Overrides:
    - ``init_tp_model_worker()``: Monkey-patches ModelRunner before worker init
    - ``handle_generate_request()``: Intercepts nnsight data from custom_logit_processor
    - ``handle_rpc_request()``: Supports returning data for nnsight RPC calls
    """

    def init_tp_model_worker(self):
        """Monkey-patch ModelRunner before the TpModelWorker is created."""
        from .model_runner import NNsightModelRunner

        import sglang.srt.model_executor.model_runner as runner_module
        original_runner = runner_module.ModelRunner
        runner_module.ModelRunner = NNsightModelRunner
        logger.info(f"NNsight: Patched ModelRunner -> {NNsightModelRunner.__name__}")
        try:
            super().init_tp_model_worker()
        finally:
            runner_module.ModelRunner = original_runner
        logger.info(f"NNsight: tp_worker.model_runner type = {type(self.tp_worker.model_runner).__name__}")

    def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
        """Override to intercept nnsight mediator data from custom_logit_processor."""
        nnsight_data = None

        # Check if custom_logit_processor carries nnsight data
        clp = recv_req.custom_logit_processor
        if clp and isinstance(clp, str) and clp.startswith("nnsight:"):
            nnsight_data = clp
            recv_req.custom_logit_processor = None  # Clear so SGLang ignores it

        result = super().handle_generate_request(recv_req)

        # Register the nnsight mediator with the model runner
        if nnsight_data is not None:
            model_runner = self.tp_worker.model_runner
            print(f"[NNsight scheduler] model_runner type: {type(model_runner).__name__}", flush=True)
            print(f"[NNsight scheduler] has nnsight_request_helper: {hasattr(model_runner, 'nnsight_request_helper')}", flush=True)
            print(f"[NNsight scheduler] nnsight_model: {model_runner.nnsight_model if hasattr(model_runner, 'nnsight_model') else 'NO ATTR'}", flush=True)
            if hasattr(model_runner, "nnsight_request_helper") and model_runner.nnsight_model is not None:
                try:
                    model_runner.nnsight_request_helper.register_request(
                        recv_req.rid, nnsight_data, model_runner.nnsight_model
                    )
                    print(f"[NNsight scheduler] Registered mediator for rid={recv_req.rid}", flush=True)
                except Exception as e:
                    print(f"[NNsight scheduler] ERROR registering mediator: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        return result

    def handle_rpc_request(self, recv_req: RpcReqInput):
        """Override to support returning data for nnsight methods."""
        if recv_req.method == "collect_nnsight":
            return self._handle_collect_nnsight(recv_req)

        return super().handle_rpc_request(recv_req)

    def _handle_collect_nnsight(self, recv_req: RpcReqInput) -> RpcReqOutput:
        """Collect saved values from the model runner and return them."""
        try:
            rids = recv_req.parameters.get("rids", [])
            model_runner = self.tp_worker.model_runner

            if not hasattr(model_runner, "nnsight_request_helper"):
                return RpcReqOutput(success=True, message="")

            saves_bytes = model_runner.nnsight_request_helper.collect_and_cleanup(
                rids, model_runner.nnsight_model
            )

            if saves_bytes:
                encoded = base64.b64encode(saves_bytes).decode("ascii")
                return RpcReqOutput(success=True, message=encoded)

            return RpcReqOutput(success=True, message="")

        except Exception as e:
            logger.error(f"Failed to collect nnsight saves: {e}")
            return RpcReqOutput(success=False, message=str(e))
