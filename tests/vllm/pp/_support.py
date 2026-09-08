"""Shared machinery for the pipeline-parallel suite.

The suite has three tiers, named by what a test needs:

* ``test_unit_*``: one process, no GPU. Ownership, the save merge, lazy
  tensors, occurrence tags, the step gate.
* ``test_harness_*``: two gloo ranks over the real ``PPInterleaver`` and
  ``PPListener``, no GPU and no vLLM engine. `Stage` opens one rank's half of
  that harness; `run_two_ranks` spawns both.
* ``test_engine_*``: real vLLM engines, skipped without the GPUs. A shared
  PP=2 engine (``conftest.pp2_engine``) carries the behavior tests; the parity
  and topology tests boot one engine per configuration in a subprocess through
  `run_worker` and compare JSON in the parent.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import tempfile
import threading
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parity_worker.py")

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = "The Eiffel Tower is located in the city of"
PROMPT_B = "Madison Square Garden is located in the city of"
# Qwen2.5-0.5B has 24 layers; at PP=2 stage 0 holds 0-11 and stage 1 holds
# 12-23, so EARLY is stage-0-owned and LATE stage-1-owned. At PP=3 the stages
# hold 0-7, 8-15, 16-23, so the three read one value from each stage.
EARLY, MIDDLE, LATE = 2, 12, 20

# A cross-stage pull that cannot complete times out after PP_PULL_TIMEOUT_S
# (30s); a scenario that finishes well inside that bound did not stall on one.
STALL_BOUND_S = 20.0


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


def free_gpus(min_free_mib: int = 12000) -> list[str]:
    """GPU indices with at least ``min_free_mib`` free, in nvidia-smi order."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        index, free = line.split(",")
        if int(free.strip()) >= min_free_mib:
            gpus.append(index.strip())
    return gpus


# ---------------------------------------------------------------------------
# Two-rank harness
# ---------------------------------------------------------------------------


def run_two_ranks(target, *args) -> None:
    """Spawn ``target(rank, world, rdv, *args)`` on two gloo ranks and join."""
    fd, rdv = tempfile.mkstemp(prefix="nnsight_pp_rdv_")
    os.close(fd)
    os.remove(rdv)
    try:
        mp.spawn(target, args=(2, rdv, *args), nprocs=2, join=True)
    finally:
        if os.path.exists(rdv):
            os.remove(rdv)


class Stage:
    """One rank's half of the two-rank harness: a listener over the world group
    and, given ``owners``, a PPInterleaver whose module map they define.

    Attributes:
        rank, world: This rank and the world size.
        listener, buffer, condition: The rank's listener and its pull buffer.
        interleaver: The PPInterleaver, or ``None`` for a wire-level test.
    """

    def __init__(self, rank: int, world: int, rdv: str, owners: Optional[dict] = None, **interleaver_kwargs: Any) -> None:
        from nnsight.modeling.vllm.pp import PPModuleMap
        from nnsight.modeling.vllm.pp_interleaver import PPInterleaver
        from nnsight.modeling.vllm.pp_listener import PPListener

        dist.init_process_group("gloo", init_method=f"file://{rdv}", rank=rank, world_size=world)
        self.rank, self.world = rank, world
        self.buffer: dict = {}
        self.condition = threading.Condition()
        self.listener = PPListener(self.buffer, self.condition, dist.group.WORLD, rank, torch.device("cpu"))
        self.listener.start()
        self.interleaver = None
        if owners is not None:
            module_map = PPModuleMap(world)
            module_map.set_derived_owners(owners)
            self.interleaver = PPInterleaver(module_map, self.listener, rank, **interleaver_kwargs)

    def mediator(self, block: str, req_id: Optional[str] = None, register: bool = True):
        """A worker for ``block`` (which reads through ``Mediator.value``), on this rank's interleaver."""
        from nnsight.intervention.interleaver import Mediator

        mediator = Mediator(compile(block, f"<pp-harness-{req_id or 'block'}>", "exec"), {"Mediator": Mediator}, {})
        if req_id is not None:
            mediator.pp_req_id = req_id
        if register:
            self.interleaver.mediators.append(mediator)
        return mediator

    def publish(self, key: tuple, value: Any) -> None:
        """Put a value in this rank's buffer and dispatch any parked pull for it."""
        with self.condition:
            self.buffer[key] = value
        self.listener.dispatch_parked(key, value)

    def close(self) -> None:
        """Stop both ranks' listen loops before the interpreter exits.

        A pending gloo recv at interpreter teardown aborts the process, so each
        rank sets its own stop flag and wakes the peer's blocked recv with a
        dummy request. Production relies on the worker process exiting hard.
        """
        from nnsight.modeling.vllm.pp_listener import REQUEST_MSG_BYTES, TAG_REQUEST

        dist.barrier()
        self.listener._stop_event.set()
        dist.send(torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8), group_dst=1 - self.rank, tag=TAG_REQUEST)
        self.listener._thread.join(timeout=5)
        dist.barrier()


# ---------------------------------------------------------------------------
# Subprocess engine runner (parity, topology)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def run_worker(scenario: str, pp: int, tp: int = 1, gpus: str = "", extra: tuple = (), prompt: str = PROMPT) -> dict:
    """Run one ``_parity_worker.py`` scenario on its own engine; return its JSON.

    Cached on its arguments, so a reference configuration boots once per
    session however many tests compare against it. Engine logs go to a file
    rather than a pipe: a pipe would make this call wait for EOF, which a
    leaked engine subprocess could hold open after the worker itself exited.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name
    log_path = output_path + ".log"
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
        cmd = [sys.executable, WORKER, scenario, "--pp", str(pp), "--tp", str(tp), "--prompt", prompt, "--output", output_path, *extra]
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=log, timeout=600, env=env, cwd=REPO_ROOT)
        try:
            with open(output_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = None
        if data is None or data.get("status") != "ok":
            with open(log_path) as log:
                tail = log.read()[-4000:]
            detail = f"{data.get('error')}\n{data.get('traceback')}" if data else f"no output written\nWORKER LOG TAIL:\n{tail}"
            raise RuntimeError(f"engine worker failed (scenario={scenario}, tp={tp}, pp={pp}, rc={result.returncode}):\n{detail}")
        return data
    finally:
        for path in (output_path, log_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def cosine(a, b) -> float:
    a = torch.tensor(a, dtype=torch.float32).flatten()
    b = torch.tensor(b, dtype=torch.float32).flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
