import base64
import pickle
from typing import List, Optional

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput


def run_nnsight_scheduler_process(
    server_args,
    port_args,
    gpu_id,
    tp_rank,
    attn_cp_rank,
    moe_dp_rank,
    moe_ep_rank,
    pp_rank,
    dp_rank,
    pipe_writer,
):
    """Custom scheduler process that monkey-patches Scheduler with NNsightScheduler.

    This function runs INSIDE the spawned subprocess. It patches the Scheduler
    class before the original run_scheduler_process creates one, ensuring our
    NNsightScheduler is used instead.
    """
    from .scheduler import NNsightScheduler

    import sglang.srt.managers.scheduler as scheduler_module
    from sglang.srt.managers.scheduler import run_scheduler_process

    original_scheduler = scheduler_module.Scheduler
    scheduler_module.Scheduler = NNsightScheduler

    print(f"[NNsight subprocess] Patched Scheduler -> {scheduler_module.Scheduler.__name__}")
    print(f"[NNsight subprocess] Scheduler is NNsightScheduler: {scheduler_module.Scheduler is NNsightScheduler}", flush=True)

    try:
        run_scheduler_process(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            pipe_writer,
        )
    finally:
        scheduler_module.Scheduler = original_scheduler


class NNsightEngine(Engine):
    """SGLang Engine subclass that injects NNsight's custom scheduler process.

    Overrides ``run_scheduler_process_func`` so the subprocess uses
    :class:`NNsightScheduler`, which creates :class:`NNsightModelRunner`
    for interleaving interventions with model forward passes.
    """

    run_scheduler_process_func = staticmethod(run_nnsight_scheduler_process)

    def __init__(self, **kwargs):
        # Force settings required for nnsight
        kwargs["disable_cuda_graph"] = True
        kwargs.setdefault("disable_overlap_schedule", True)
        # Enable custom_logit_processor so we can transport mediator data
        kwargs["enable_custom_logit_processor"] = True

        print(f"NNsightEngine.__init__: run_scheduler_process_func = {self.run_scheduler_process_func}")
        print(f"NNsightEngine.__init__: is our func: {self.run_scheduler_process_func is run_nnsight_scheduler_process}")

        super().__init__(**kwargs)

    def collect_nnsight_saves(
        self, rids: List[str]
    ) -> Optional[bytes]:
        """Collect saved intervention values from the scheduler subprocess.

        Sends an RPC request to the scheduler's ``collect_nnsight`` method
        and returns the pickled saves bytes from the response message.
        """
        if self.send_to_rpc is None:
            return None

        obj = RpcReqInput(
            method="collect_nnsight",
            parameters={"rids": rids},
        )
        self.send_to_rpc.send_pyobj(obj)

        import zmq
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)

        if recv_req.success and recv_req.message:
            try:
                return base64.b64decode(recv_req.message)
            except Exception:
                return None
        return None
