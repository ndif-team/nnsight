"""The pull buffer holds exactly the locations workers wait on: a run over
many modules buffers one entry for the one location the block reads."""

import os
import threading

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MODULES = 4
STEPS = 3


def run(rank: int, world: int, rdv: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rdv}", rank=rank, world_size=world
    )
    from nnsight.intervention.interleaver import Mediator
    from nnsight.modeling.vllm.pp import PPModuleMap
    from nnsight.modeling.vllm.pp_interleaver import PPInterleaver
    from nnsight.modeling.vllm.pp_listener import PPListener

    module_map = PPModuleMap(world)
    module_map.set_derived_owners(
        {f"h.{m}": 0 for m in range(MODULES)} | {"peer": 1}
    )

    buffer: dict = {}
    condition = threading.Condition()
    listener = PPListener(
        buffer, condition, dist.group.WORLD, rank, torch.device("cpu")
    )
    listener.start()
    interleaver = PPInterleaver(module_map, listener, rank)

    block = 'x = Mediator.value("model.h.0.output")\n'
    mediator = Mediator(
        compile(block, "<publish-scope>", "exec"), {"Mediator": Mediator}, {}
    )
    interleaver.mediators.append(mediator)

    if rank == 0:
        value = torch.arange(4.0)
        with interleaver:
            for step in range(STEPS):
                for m in range(MODULES):
                    interleaver.handle(f"model.h.{m}.input", ((value,), {}))
                    interleaver.handle(f"model.h.{m}.output", value)
        assert list(buffer) == [("model.h.0.output.i0", None)], list(buffer)
        assert torch.equal(buffer[("model.h.0.output.i0", None)], value)
        print("rank 0: buffered 1 of", STEPS * MODULES * 2, "handled locations")

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


def test_publish_covers_only_waited_locations(tmp_path=None):
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_publish_scope_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(run, args=(2, rdv), nprocs=2, join=True)


if __name__ == "__main__":
    test_publish_covers_only_waited_locations()
    print("OK")
