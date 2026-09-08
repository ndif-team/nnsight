"""Two-process gloo round-trip check of the ported pp_listener (0.8, split-phase).

Rank 1 is the producer: starts a listener, publishes values into its buffer.
Rank 0 is the consumer: issues split-phase pulls and completes them.

Covers: buffered serve, parked-then-dispatched serve, tuple round-trip, int32
dtype preservation, error reply for a non-tensor value, abandoned-pull error
reply on scoped clear, drain barrier.
"""
import os
import threading
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp




def run(rank: int, world: int, rdv: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rdv}", rank=rank, world_size=world
    )
    from nnsight.modeling.vllm.pp_listener import PPListener

    buffer: dict = {}
    condition = threading.Condition()
    listener = PPListener(
        buffer, condition, dist.group.WORLD, rank, torch.device("cpu")
    )
    listener.start()

    if rank == 1:
        # Producer: publish one value up front, one late, one non-tensor.
        with condition:
            buffer[("model.h.0.output.i0", "req-a")] = torch.arange(
                12, dtype=torch.float32
            ).reshape(3, 4)
            buffer[("model.h.0.output.i0", None)] = (
                torch.ones(2, 2),
                torch.full((2, 2), 5.0),
            )
            buffer[("model.samples.i0", "req-a")] = torch.tensor(
                [7, 8, 9], dtype=torch.int32
            )
            buffer[("model.h.1.inputs.i0", "req-a")] = {"not": "a tensor"}
        # The late value arrives after a delay; dispatch_parked serves the
        # consumer's already-parked pull.
        def publish_late():
            time.sleep(0.5)
            key = ("model.h.2.output.i0", "req-a")
            with condition:
                buffer[key] = torch.full((2, 3), 2.5)
            listener.dispatch_parked(key, buffer[key])

        threading.Thread(target=publish_late).start()
        listener.drain_barrier()
        # After the barrier, scoped-clear a finished request while a pull for a
        # never-produced value is parked; it must get an error reply.
        time.sleep(0.5)
        listener.clear_buffer(req_ids=["req-a"])
        time.sleep(1.0)
        listener.stop()
    else:
        # Consumer: several concurrent split-phase pulls, issued before any wait.
        p1 = listener.begin_pull(1, "model.h.0.output.i0", "req-a")
        p2 = listener.begin_pull(1, "model.h.0.output.i0", None)
        p3 = listener.begin_pull(1, "model.samples.i0", "req-a")
        p4 = listener.begin_pull(1, "model.h.2.output.i0", "req-a")  # parked
        v1 = p1.complete()
        assert v1.shape == (3, 4) and v1[2, 3] == 11, v1
        v2 = p2.complete()
        assert isinstance(v2, tuple) and torch.all(v2[1] == 5.0), v2
        v3 = p3.complete()
        assert v3.dtype == torch.int32 and v3.tolist() == [7, 8, 9], v3
        v4 = p4.complete()
        assert v4.shape == (2, 3) and torch.all(v4 == 2.5), v4
        # Error reply: a dict-valued read cannot be wire-encoded.
        p5 = listener.begin_pull(1, "model.h.1.inputs.i0", "req-a")
        try:
            p5.complete()
            raise AssertionError("dict pull should have raised")
        except RuntimeError as e:
            assert "model.h.1" in str(e) and "detach" in str(e), e
        listener.drain_barrier()
        # A pull parked for a value that is never produced, then the request is
        # finalized: the scoped clear error-replies it.
        p6 = listener.begin_pull(1, "model.h.9.output.i0", "req-a")
        try:
            p6.complete(timeout=5.0)
            raise AssertionError("abandoned pull should have raised")
        except RuntimeError as e:
            assert "never produced" in str(e), e
        print("consumer: all pull scenarios passed")
        listener.stop()

    dist.barrier()
    # Drain the listen loops before exit: a pending gloo recv at interpreter
    # teardown SIGABRTs. Set our own stop flag, then wake the PEER's blocked
    # recv with a dummy request (its loop errors on the empty key, sees the
    # stop flag, and returns). Production doesn't need this: the daemon
    # listener dies with the (hard-exiting) vLLM worker process.
    from nnsight.modeling.vllm.pp_listener import REQUEST_MSG_BYTES, TAG_REQUEST
    listener._stop_event.set()
    dist.send(
        torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8),
        group_dst=1 - rank, tag=TAG_REQUEST,
    )
    listener._thread.join(timeout=5)
    dist.barrier()


def test_listener_protocol(tmp_path=None):
    """Round-trip the full pull protocol across two spawned gloo ranks."""
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_pp_listener_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(run, args=(2, rdv), nprocs=2, join=True)


if __name__ == "__main__":
    test_listener_protocol()
    print("OK")
