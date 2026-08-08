"""Two-rank cross-stage interleaving through the real PP machinery, no vLLM.

Rank 0 owns ``model.h.0``; rank 1 owns ``model.h.1``. Both ranks run the SAME
intervention block (replicated, as in real PP): it reads both modules'
outputs, combines them, and saves the result. On each rank one read is local
(parks, served by that rank's forward) and one is remote (answered with a
LazyRemoteTensor; forcing it parks on a pull the serve point completes from
the peer's published buffer). Both ranks must finish with the same value.

Also covers: the published buffer serving a repeat pull of the same visit.
"""

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

    # The replicated block, identical on both ranks. Reads go through
    # Mediator.value exactly as an eproperty's __get__ would. Both remote
    # reads come back as lazies; the combine line forces each rank's remote
    # one, parking the worker on a pull the serve point completes. Ordering
    # note: the force comes AFTER both reads — a force before a local read
    # would park the worker past its local visit, the same local-after-remote
    # ordering error the 0.7 branch raises for.
    # Operand order matters on the non-owning rank: the lazy must lead the
    # expression (its own __add__ runs, plain Python) — with a real tensor on
    # the left, torch's override protocol dispatches instead, and forcing a
    # value inside that dispatch is forbidden (see the __torch_function__
    # guard in lazy_remote_tensor.py).
    block = """
a = Mediator.value("model.h.0.output")
b = Mediator.value("model.h.1.output")
c = (b + a * 2).sum().item()
"""
    lcls: dict = {}
    mediator = Mediator(
        compile(block, "<pp-test>", "exec"), {"Mediator": Mediator}, lcls
    )
    interleaver.mediators.append(mediator)

    # This rank's "forward": produce only the module this stage owns. The
    # owner's handle serves the local park, applies the swap, and publishes
    # the post-intervention rows for the peer's pull.
    values = {
        0: ("model.h.0.output", torch.arange(4, dtype=torch.float32)),
        1: ("model.h.1.output", torch.full((4,), 5.0)),
    }
    provider, value = values[rank]
    with interleaver:
        interleaver.handle(provider, value)
    interleaver.serve_pulls()

    assert not mediator.alive, "block should have run to completion"
    # c = sum(a*2 + b) = sum([0,2,4,6] + [5,5,5,5]) = 12 + 20 = 32 on BOTH
    # ranks: the pulled value must equal the peer's local one.
    assert mediator.lcls["c"] == 32.0, dict(mediator.lcls)
    # The published buffer serves repeat pulls of the same visit.
    if rank == 1:
        pulled = listener.begin_pull(0, "model.h.0.output.i0").complete()
        assert torch.equal(pulled, torch.arange(4, dtype=torch.float32)), pulled

    listener.drain_barrier()
    result = mediator.lcls["c"]
    print(f"rank {rank}: cross-stage block ok (c={result})")

    # Drain the listen loops before exit (a pending gloo recv at interpreter
    # teardown SIGABRTs); production relies on hard worker-process exit.
    from nnsight.modeling.vllm.pp_listener import REQUEST_MSG_BYTES, TAG_REQUEST

    dist.barrier()
    listener._stop_event.set()
    dist.send(
        torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8),
        group_dst=1 - rank, tag=TAG_REQUEST,
    )
    listener._thread.join(timeout=5)
    dist.barrier()


def test_pp_interleaver_crossstage(tmp_path=None):
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_pp_interleaver_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(run, args=(2, rdv), nprocs=2, join=True)


if __name__ == "__main__":
    test_pp_interleaver_crossstage()
    print("OK")
