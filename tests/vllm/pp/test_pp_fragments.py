"""Fragments (within-stage TP gather) riding the PP interleaver.

Rank 0 owns ``model.h.0``, whose output is marked fragmented by a fake
Fragments (whole = piece * 2, fragment = whole / 2 — the arithmetic of an
all-reduce over two identical partials, with no real collective). Rank 1 owns
``model.h.1``, unfragmented. Both ranks run the same block reading both
outputs.

Three things must hold on the fragmented location:
* the local worker reads the assembled whole,
* the peer's pull receives that same whole (publish runs inside handle's
  gather bracket, before the re-split),
* the model's own forward gets the re-split piece back from ``handle``.
"""

import os
import threading

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

FRAGMENTED = "model.h.0.output"


def run(rank: int, world: int, rdv: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rdv}", rank=rank, world_size=world
    )
    from nnsight.intervention.fragments import Fragments
    from nnsight.intervention.interleaver import Mediator
    from nnsight.modeling.vllm.pp import PPModuleMap
    from nnsight.modeling.vllm.pp_interleaver import PPInterleaver
    from nnsight.modeling.vllm.pp_listener import PPListener

    class FakeFragments(Fragments):
        enabled = True

        def fragmented(self, location: str) -> bool:
            return location == FRAGMENTED

        def whole(self, location: str, value: torch.Tensor) -> torch.Tensor:
            return value * 2

        def fragment(self, location: str, whole: torch.Tensor) -> torch.Tensor:
            return whole / 2

    module_map = PPModuleMap(world)
    module_map.set_derived_owners({"h.0": 0, "h.1": 1})

    buffer: dict = {}
    condition = threading.Condition()
    listener = PPListener(
        buffer, condition, dist.group.WORLD, rank, torch.device("cpu")
    )
    listener.start()

    interleaver = PPInterleaver(
        module_map, listener, rank, fragments=FakeFragments()
    )

    block = """
a = Mediator.value("model.h.0.output")
b = Mediator.value("model.h.1.output")
c = (a + b).sum().item()
"""
    lcls: dict = {}
    mediator = Mediator(
        compile(block, "<pp-fragments-test>", "exec"), {"Mediator": Mediator}, lcls
    )
    interleaver.mediators.append(mediator)

    # This rank's "forward". Rank 0 serves the fragmented location with its
    # piece [0,1,2,3]; workers everywhere must see the whole [0,2,4,6].
    values = {
        0: (FRAGMENTED, torch.arange(4, dtype=torch.float32)),
        1: ("model.h.1.output", torch.full((4,), 5.0)),
    }
    provider, piece = values[rank]
    with interleaver:
        returned = interleaver.handle(provider, piece.clone())
    interleaver.serve_pulls()

    assert not mediator.alive, "block should have run to completion"
    # c = sum(whole + b) = sum([0,2,4,6] + [5,5,5,5]) = 32 on BOTH ranks.
    assert mediator.lcls["c"] == 32.0, dict(mediator.lcls)

    if rank == 0:
        # The model's forward carries on from the re-split piece.
        assert torch.equal(returned, piece), returned
    if rank == 1:
        # The buffer holds the whole, not the producing rank's piece.
        pulled = listener.begin_pull(0, f"{FRAGMENTED}.i0").complete()
        assert torch.equal(
            pulled, torch.tensor([0.0, 2.0, 4.0, 6.0])
        ), pulled

    listener.drain_barrier()
    print(f"rank {rank}: fragments-through-PP ok (c={mediator.lcls['c']})")

    from nnsight.modeling.vllm.pp_listener import REQUEST_MSG_BYTES, TAG_REQUEST

    dist.barrier()
    listener._stop_event.set()
    dist.send(
        torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8),
        group_dst=1 - rank, tag=TAG_REQUEST,
    )
    listener._thread.join(timeout=5)
    dist.barrier()


def test_pp_fragments(tmp_path=None):
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_pp_fragments_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(run, args=(2, rdv), nprocs=2, join=True)


if __name__ == "__main__":
    test_pp_fragments()
    print("OK")
