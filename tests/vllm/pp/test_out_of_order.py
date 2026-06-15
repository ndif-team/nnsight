"""PP out-of-forward-order access must raise, not hang.

Single-GPU nnsight rejects accessing a later module before an earlier one within
one forward with ``OutOfOrderError`` (a ``MissedProviderError``). Under PP this
same pattern — reading a downstream (later-stage) module before a local
(this-stage) one inside ``tracer.iter`` — previously deadlocked: stage 0's
downstream access released its forward early (``past_local`` + ``go_remote``), so
the subsequent local hook was missed and could only be served by the next
forward, which was itself gated waiting for the mediator to advance.

The PP path must surface the SAME ``OutOfOrderError`` promptly. This test runs
the out-of-order scenario at PP=2 in a subprocess and asserts it reports an
out-of-order error within a bounded time (i.e. does not hang).
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest


def _find_free_gpus(min_free_mib=4000):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        free = []
        for line in result.stdout.strip().split("\n"):
            idx, free_mib = line.split(",")
            if int(free_mib.strip()) >= min_free_mib:
                free.append(int(idx.strip()))
        return free
    except Exception:
        return []


FREE_GPUS = _find_free_gpus(min_free_mib=4000)

if len(FREE_GPUS) < 2:
    pytest.skip(
        f"PP out-of-order test needs 2 free GPUs, found {len(FREE_GPUS)}: {FREE_GPUS}",
        allow_module_level=True,
    )

GPU_PP2 = f"{FREE_GPUS[0]},{FREE_GPUS[1]}"
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "manual", "_pp_worker.py")
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)

# Generous enough for model load + a few decode steps; a genuine deadlock blows
# past it. The fix makes the worker raise in well under this.
WORKER_TIMEOUT_S = 180


def _run_worker_raw(cuda_visible_devices, scenario, extra_args):
    """Run a worker scenario; return its parsed JSON result dict.

    Raises ``subprocess.TimeoutExpired`` if the worker hangs (the pre-fix
    behavior), or ``RuntimeError`` if the subprocess dies without writing output.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        cmd = [sys.executable, WORKER_SCRIPT, scenario, "--output", output_path, *extra_args]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S,
            env=env, cwd=REPO_ROOT,
        )
        with open(output_path) as f:
            content = f.read()
        if not content:
            raise RuntimeError(
                f"Worker wrote no output (rc={result.returncode}):\n"
                f"STDERR (last 3000):\n{result.stderr[-3000:]}"
            )
        return json.loads(content)
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def test_pp2_out_of_order_iter_raises_not_hangs():
    """Reading downstream logits before a local layer each iter raises
    OutOfOrderError (matching single-GPU), instead of deadlocking."""
    try:
        data = _run_worker_raw(
            GPU_PP2, "multigen_ooo",
            ["--pp", "2", "--prompt", "Madison Square Garden is in the city of",
             "--max_tokens", "3"],
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"PP=2 out-of-order trace HUNG (no result within {WORKER_TIMEOUT_S}s) — "
            "the deadlock is not fixed; it must raise OutOfOrderError promptly."
        )

    assert data.get("status") == "error", (
        f"Out-of-order access should raise, but the worker returned: {data!r}"
    )
    msg = f"{data.get('error', '')}\n{data.get('traceback', '')}".lower()
    assert "outofordererror" in msg, (
        f"Expected an OutOfOrderError, got:\n{data.get('error')}\n{data.get('traceback')}"
    )
    assert "forward-pass order" in msg, (
        "OutOfOrderError should explain the forward-pass-order contract; got:\n"
        f"{data.get('error')}"
    )
