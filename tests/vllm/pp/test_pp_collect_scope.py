"""serve_pulls scoped to a request set: only the named requests' workers are
resumed; the rest keep their parks for their own step serves."""

import os
import threading

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank: int, world: int, rdv: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rdv}", rank=rank, world_size=world
    )
    from nnsight.intervention.interleaver import Mediator
    from nnsight.modeling.vllm.pp import PPModuleMap
    from nnsight.modeling.vllm.pp_interleaver import PPInterleaver
    from nnsight.modeling.vllm.pp_listener import PPListener

    module_map = PPModuleMap(world)
    module_map.set_derived_owners({"h.0": 0, "h.1": 1})

    buffer: dict = {}
    condition = threading.Condition()
    listener = PPListener(
        buffer, condition, dist.group.WORLD, rank, torch.device("cpu")
    )
    listener.start()

    interleaver = PPInterleaver(module_map, listener, rank)

    block = """
a = Mediator.value("model.h.1.output")
result = float(a.sum())
"""
    mediators = {}
    for req_id in ("req-A", "req-B"):
        mediator = Mediator(
            compile(block, f"<{req_id}>", "exec"), {"Mediator": Mediator}, {}
        )
        mediator.pp_req_id = req_id
        mediators[req_id] = mediator
        interleaver.mediators.append(mediator)

    with interleaver:
        if rank == 0:
            interleaver.handle("model.h.0.output", torch.zeros(4))
        else:
            interleaver.handle("model.h.1.output", torch.arange(4.0))

    if rank == 0:
        # Both workers are parked on a pull of the peer's published value.
        interleaver.serve_pulls(block=True, only={"req-A"})
        assert not mediators["req-A"].alive, "the named request's worker finishes"
        assert mediators["req-A"].lcls["result"] == 6.0
        assert mediators["req-B"].alive, "the other request's worker keeps its park"
        assert mediators["req-B"].pending is not None

        interleaver.serve_pulls(block=True)
        assert not mediators["req-B"].alive
        assert mediators["req-B"].lcls["result"] == 6.0
        print("rank 0: scoped serve ok")

    listener.drain_barrier()

    from nnsight.modeling.vllm.pp_listener import REQUEST_MSG_BYTES, TAG_REQUEST

    dist.barrier()
    listener._stop_event.set()
    dist.send(
        torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8),
        group_dst=1 - rank, tag=TAG_REQUEST,
    )
    listener._thread.join(timeout=5)
    dist.barrier()


def test_serve_pulls_scoped_to_requests(tmp_path=None):
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_pp_collect_scope_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(run, args=(2, rdv), nprocs=2, join=True)



def run_finalize(rank: int, world: int, rdv: str) -> None:
    import time

    dist.init_process_group(
        "gloo", init_method=f"file://{rdv}", rank=rank, world_size=world
    )
    from nnsight.intervention.interleaver import Mediator
    from nnsight.modeling.vllm.pp import PPModuleMap
    from nnsight.modeling.vllm.pp_interleaver import PPInterleaver
    from nnsight.modeling.vllm.pp_listener import PPListener

    module_map = PPModuleMap(world)
    module_map.set_derived_owners({"h.0": 0, "h.1": 1})

    buffer: dict = {}
    condition = threading.Condition()
    listener = PPListener(
        buffer, condition, dist.group.WORLD, rank, torch.device("cpu")
    )
    listener.start()
    interleaver = PPInterleaver(module_map, listener, rank)

    block = """
a = Mediator.value("model.h.1.output")
result = float(a.sum())
"""
    producible = Mediator(
        compile(block, "<producible>", "exec"), {"Mediator": Mediator}, {}
    )
    producible.pp_req_id = "req-P"
    # A pull for a round the pipeline has not produced: the request has no
    # completed rounds and the owner publishes nothing for it.
    lookahead = Mediator(
        compile(block, "<lookahead>", "exec"), {"Mediator": Mediator}, {}
    )
    lookahead.pp_req_id = "req-L"
    interleaver.rounds.update({"req-P": 1, "req-L": 0})
    interleaver.mediators.append(producible)
    if rank == 0:
        interleaver.mediators.append(lookahead)

    with interleaver:
        if rank == 0:
            interleaver.handle("model.h.0.output", torch.zeros(4))
        else:
            interleaver.handle("model.h.1.output", torch.arange(4.0))

    if rank == 0:
        # A later step's scheduling has rebuilt the per-step list; the
        # finished workers are named explicitly at collect.
        interleaver.mediators.clear()
        t0 = time.time()
        interleaver.serve_pulls(
            block=True, drain=False, mediators=[producible, lookahead]
        )
        elapsed = time.time() - t0
        assert elapsed < 10, f"finalize serve took {elapsed:.1f}s"
        assert not producible.alive, "the produced round's pull completes"
        assert producible.lcls["result"] == 6.0
        assert lookahead.alive, "a pull past generation stays parked"
        assert lookahead.pending is not None

        interleaver.discard_pulls(lookahead)
        assert not any(
            key[0] == id(lookahead) for key in interleaver._pulls
        ), "a released worker leaves no pull records"
        print(f"rank 0: finalize serve ok in {elapsed:.2f}s")

    listener.drain_barrier()

    from nnsight.modeling.vllm.pp_listener import REQUEST_MSG_BYTES, TAG_REQUEST

    dist.barrier()
    listener._stop_event.set()
    dist.send(
        torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8),
        group_dst=1 - rank, tag=TAG_REQUEST,
    )
    listener._thread.join(timeout=5)
    dist.barrier()


def test_finalize_serves_finished_workers_by_round(tmp_path=None):
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_pp_finalize_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(run_finalize, args=(2, rdv), nprocs=2, join=True)


if __name__ == "__main__":
    test_serve_pulls_scoped_to_requests()
    test_finalize_serves_finished_workers_by_round()
    print("OK")
