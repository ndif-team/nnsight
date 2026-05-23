#!/usr/bin/env python
"""
Profiling and corner-case testing for Pipeline Parallelism (PP) in NNsight's vLLM integration.

Standalone script (not pytest). Profiles PP overhead and tests corner cases.

Part 1 (A-G): Overhead profiling comparing PP=1 vs PP=2 with GPT-2.
Part 2 (H-L): Corner case tests for LazyRemoteTensor and PP edge cases.

Usage:
    conda run -n ndif-dev python tests/test_vllm_pp_profile.py

Requires: 2+ NVIDIA GPUs with >= 4 GB free each.

GPT-2 has 12 layers. With PP=2:
  Stage 0 (rank 0): layers 0-5, embed (wte, wpe)
  Stage 1 (rank 1): layers 6-11, ln_f, lm_head, logits, samples
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _find_free_gpus(min_free_mib: int = 4000) -> List[int]:
    """Return indices of GPUs with at least min_free_mib free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        free = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            idx, free_mib = int(parts[0].strip()), int(parts[1].strip())
            if free_mib >= min_free_mib:
                free.append(idx)
        return free
    except Exception:
        return []


def check_gpus():
    """Check that we have enough GPUs. Exit early if not."""
    count = torch.cuda.device_count()
    if count < 2:
        print(f"SKIP: Need >= 2 CUDA GPUs, found {count}.")
        sys.exit(0)

    free = _find_free_gpus(min_free_mib=4000)
    if len(free) < 3:
        print(
            f"SKIP: Need 3 free GPUs (1 for PP=1, 2 for PP=2), "
            f"found {len(free)} free: {free}"
        )
        sys.exit(0)

    return free


# ---------------------------------------------------------------------------
# Subprocess worker execution (reuses _pp_worker.py pattern)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pp_profile_worker.py")


def _run_worker(
    cuda_visible_devices: str,
    scenario: str,
    pp: int,
    prompt: str = "The Eiffel Tower is located in the city of",
    extra_args: Optional[List[str]] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Run a profiling scenario in a subprocess. Returns parsed JSON output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cmd = [
            sys.executable,
            PROFILE_WORKER,
            scenario,
            "--pp", str(pp),
            "--prompt", prompt,
            "--output", output_path,
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
            cwd=REPO_ROOT,
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Worker failed (rc={result.returncode})",
                "stderr": result.stderr[-4000:] if result.stderr else "(empty)",
            }

        with open(output_path, "r") as f:
            data = json.load(f)

        return data

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def fmt_ms(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val * 1000:.2f} ms"


def fmt_overhead(pp1_mean: Optional[float], pp2_mean: Optional[float]) -> str:
    if pp1_mean is None or pp2_mean is None or pp1_mean == 0:
        return "N/A"
    pct = (pp2_mean - pp1_mean) / pp1_mean * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def print_header(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_timing_row(label: str, pp1_data: Dict, pp2_data: Dict, key: str = "times"):
    """Print a timing comparison row."""
    pp1_ok = pp1_data.get("status") == "ok"
    pp2_ok = pp2_data.get("status") == "ok"

    if pp1_ok:
        pp1_times = pp1_data[key]
        pp1_mean = statistics.mean(pp1_times)
        pp1_std = statistics.stdev(pp1_times) if len(pp1_times) > 1 else 0.0
    else:
        pp1_mean = pp1_std = None

    if pp2_ok:
        pp2_times = pp2_data[key]
        pp2_mean = statistics.mean(pp2_times)
        pp2_std = statistics.stdev(pp2_times) if len(pp2_times) > 1 else 0.0
    else:
        pp2_mean = pp2_std = None

    overhead = fmt_overhead(pp1_mean, pp2_mean)

    print(f"  {label:<45s}  "
          f"PP=1: {fmt_ms(pp1_mean):>10s} +/- {fmt_ms(pp1_std):>8s}  |  "
          f"PP=2: {fmt_ms(pp2_mean):>10s} +/- {fmt_ms(pp2_std):>8s}  |  "
          f"Overhead: {overhead:>8s}")

    if not pp1_ok:
        print(f"    PP=1 ERROR: {pp1_data.get('error', 'unknown')[:200]}")
    if not pp2_ok:
        print(f"    PP=2 ERROR: {pp2_data.get('error', 'unknown')[:200]}")


def print_corner_case(label: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  -- {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Part 2: Corner case tests (run locally, no vLLM needed)
# ---------------------------------------------------------------------------

def test_lazy_no_materialization():
    """Test H: Operations that should NOT trigger materialization."""
    from nnsight.modeling.vllm.lazy_remote_tensor import LazyRemoteTensor

    results = []

    def make_lazy():
        return LazyRemoteTensor(
            source_rank=1,
            provider_string="model.layers.50.output.i0",
            shape=(1, 5, 768),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    # H1: .shape
    lazy = make_lazy()
    _ = lazy.shape
    results.append(("lazy.shape", lazy._real is None))

    # H2: .dtype
    lazy = make_lazy()
    _ = lazy.dtype
    results.append(("lazy.dtype", lazy._real is None))

    # H3: .device
    lazy = make_lazy()
    _ = lazy.device
    results.append(("lazy.device", lazy._real is None))

    # H4: __setitem__ (write is no-op)
    lazy = make_lazy()
    lazy[:] = torch.zeros(1, 5, 768)
    results.append(("lazy[:] = zeros", lazy._real is None))

    # H5: chained indexing write
    lazy = make_lazy()
    lazy[0][:] = torch.zeros(5, 768)
    results.append(("lazy[0][:] = zeros", lazy._real is None))

    # H6: .save() is no-op
    lazy = make_lazy()
    ret = lazy.save()
    results.append(("lazy.save()", lazy._real is None and ret is lazy))

    # H7: __repr__ without materialization
    lazy = make_lazy()
    _ = repr(lazy)
    results.append(("repr(lazy)", lazy._real is None))

    # H8: __getitem__ returns self
    lazy = make_lazy()
    result = lazy[0]
    results.append(("lazy[0] returns self", result is lazy and lazy._real is None))

    return results


def test_lazy_materialization():
    """Test I: Operations that SHOULD trigger materialization."""
    from nnsight.modeling.vllm.lazy_remote_tensor import LazyRemoteTensor

    results = []

    def make_lazy_with_real():
        """Create a lazy tensor with _real pre-set (simulates post-pull state)."""
        lazy = LazyRemoteTensor(
            source_rank=1,
            provider_string="model.layers.50.output.i0",
            shape=(1, 5, 768),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        real = torch.randn(1, 5, 768)
        lazy._real = real
        return lazy, real

    def make_lazy_with_pull():
        """Create a lazy tensor with a _pull_fn that sets _real."""
        lazy = LazyRemoteTensor(
            source_rank=1,
            provider_string="model.layers.50.output.i0",
            shape=(1, 5, 768),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        real = torch.randn(1, 5, 768)
        lazy._pull_fn = lambda src, key: real
        return lazy, real

    # I1: __add__ (with pre-set _real)
    lazy, real = make_lazy_with_real()
    try:
        result = lazy + 1
        ok = isinstance(result, torch.Tensor) and torch.allclose(result, real + 1)
        results.append(("lazy + 1 (pre-set _real)", ok))
    except Exception as e:
        results.append(("lazy + 1 (pre-set _real)", False, str(e)))

    # I2: __mul__ (with pre-set _real)
    lazy, real = make_lazy_with_real()
    try:
        result = lazy * 2
        ok = isinstance(result, torch.Tensor) and torch.allclose(result, real * 2)
        results.append(("lazy * 2 (pre-set _real)", ok))
    except Exception as e:
        results.append(("lazy * 2 (pre-set _real)", False, str(e)))

    # I3: torch.sum (with pre-set _real, uses __torch_function__)
    lazy, real = make_lazy_with_real()
    try:
        result = torch.sum(lazy)
        ok = isinstance(result, torch.Tensor) and torch.allclose(result, torch.sum(real))
        results.append(("torch.sum(lazy) (pre-set _real)", ok))
    except Exception as e:
        results.append(("torch.sum(lazy) (pre-set _real)", False, str(e)))

    # I4: __add__ with _pull_fn (simulates real materialization path)
    lazy, real = make_lazy_with_pull()
    try:
        assert lazy._real is None, "Should start unmaterialized"
        result = lazy + 1
        ok = (
            lazy._real is not None
            and isinstance(result, torch.Tensor)
            and torch.allclose(result, real + 1)
        )
        results.append(("lazy + 1 (via _pull_fn)", ok))
    except Exception as e:
        results.append(("lazy + 1 (via _pull_fn)", False, str(e)))

    # I5: __mul__ with _pull_fn
    lazy, real = make_lazy_with_pull()
    try:
        result = lazy * 2
        ok = lazy._real is not None and torch.allclose(result, real * 2)
        results.append(("lazy * 2 (via _pull_fn)", ok))
    except Exception as e:
        results.append(("lazy * 2 (via _pull_fn)", False, str(e)))

    # I6: torch.sum with _pull_fn
    lazy, real = make_lazy_with_pull()
    try:
        result = torch.sum(lazy)
        ok = lazy._real is not None and torch.allclose(result, torch.sum(real))
        results.append(("torch.sum(lazy) (via _pull_fn)", ok))
    except Exception as e:
        results.append(("torch.sum(lazy) (via _pull_fn)", False, str(e)))

    # I7: .mean() method — may or may not work depending on __torch_function__ dispatch
    lazy, real = make_lazy_with_pull()
    try:
        result = lazy.mean()
        ok = lazy._real is not None and torch.allclose(result, real.mean())
        results.append(("lazy.mean() (via _pull_fn)", ok))
    except AttributeError:
        results.append(("lazy.mean() (via _pull_fn)", False,
                         "AttributeError: LazyRemoteTensor has no .mean() — "
                         "needs explicit delegation or __getattr__ fallback"))
    except Exception as e:
        results.append(("lazy.mean() (via _pull_fn)", False, str(e)))

    # I8: __sub__ with _pull_fn
    lazy, real = make_lazy_with_pull()
    try:
        result = lazy - 1
        ok = lazy._real is not None and torch.allclose(result, real - 1)
        results.append(("lazy - 1 (via _pull_fn)", ok))
    except Exception as e:
        results.append(("lazy - 1 (via _pull_fn)", False, str(e)))

    # I9: __neg__ with _pull_fn
    lazy, real = make_lazy_with_pull()
    try:
        result = -lazy
        ok = lazy._real is not None and torch.allclose(result, -real)
        results.append(("-lazy (via _pull_fn)", ok))
    except Exception as e:
        results.append(("-lazy (via _pull_fn)", False, str(e)))

    # I10: __matmul__ with _pull_fn
    lazy_small = LazyRemoteTensor(
        source_rank=1,
        provider_string="test.matmul",
        shape=(3, 4),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    real_small = torch.randn(3, 4)
    lazy_small._pull_fn = lambda src, key: real_small
    try:
        other = torch.randn(4, 5)
        result = lazy_small @ other
        ok = lazy_small._real is not None and torch.allclose(result, real_small @ other)
        results.append(("lazy @ other (via _pull_fn)", ok))
    except Exception as e:
        results.append(("lazy @ other (via _pull_fn)", False, str(e)))

    # I11: No _pull_fn set — should raise RuntimeError
    lazy_no_pull = LazyRemoteTensor(
        source_rank=1,
        provider_string="test.no_pull",
        shape=(2, 3),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    try:
        _ = lazy_no_pull + 1
        results.append(("lazy + 1 (no _pull_fn)", False, "Should have raised RuntimeError"))
    except RuntimeError as e:
        ok = "no pull function" in str(e).lower()
        results.append(("lazy + 1 (no _pull_fn) -> RuntimeError", ok))
    except Exception as e:
        results.append(("lazy + 1 (no _pull_fn)", False, f"Unexpected: {type(e).__name__}: {e}"))

    return results


def test_pp_listener_corner_cases():
    """Corner cases for PPListener (no vLLM needed)."""
    import threading
    from nnsight.modeling.vllm.pp_listener import PPListener

    results = []

    # Immediate lookup
    buf = {"a.output.i0": torch.randn(2, 3)}
    cond = threading.Condition()
    listener = PPListener(buf, cond)
    try:
        val = listener.local_lookup("a.output.i0", timeout=1.0)
        results.append(("PPListener immediate lookup", torch.equal(val, buf["a.output.i0"])))
    except Exception as e:
        results.append(("PPListener immediate lookup", False, str(e)))

    # Timeout
    buf2 = {}
    cond2 = threading.Condition()
    listener2 = PPListener(buf2, cond2)
    try:
        listener2.local_lookup("missing.key", timeout=0.1)
        results.append(("PPListener timeout", False, "Should have raised TimeoutError"))
    except TimeoutError:
        results.append(("PPListener timeout -> TimeoutError", True))
    except Exception as e:
        results.append(("PPListener timeout", False, str(e)))

    # Async value arrival
    buf3 = {}
    cond3 = threading.Condition()
    listener3 = PPListener(buf3, cond3)
    result_holder = [None]
    error_holder = [None]

    def async_lookup():
        try:
            result_holder[0] = listener3.local_lookup("delayed.key", timeout=5.0)
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=async_lookup)
    t.start()

    time.sleep(0.1)  # let the thread start waiting
    tensor = torch.randn(4, 5)
    with cond3:
        buf3["delayed.key"] = tensor
        cond3.notify_all()

    t.join(timeout=5.0)
    if error_holder[0] is not None:
        results.append(("PPListener async arrival", False, str(error_holder[0])))
    elif result_holder[0] is not None:
        results.append(("PPListener async arrival", torch.equal(result_holder[0], tensor)))
    else:
        results.append(("PPListener async arrival", False, "Thread did not complete"))

    # Stop unblocks waiters
    buf4 = {}
    cond4 = threading.Condition()
    listener4 = PPListener(buf4, cond4)
    error_holder2 = [None]

    def blocking_lookup():
        try:
            listener4.local_lookup("never.arriving", timeout=30.0)
        except RuntimeError as e:
            error_holder2[0] = e
        except Exception as e:
            error_holder2[0] = e

    t2 = threading.Thread(target=blocking_lookup)
    t2.start()
    time.sleep(0.1)
    listener4.stop()
    t2.join(timeout=5.0)

    if isinstance(error_holder2[0], RuntimeError) and "stopped" in str(error_holder2[0]).lower():
        results.append(("PPListener stop unblocks waiter", True))
    else:
        results.append(("PPListener stop unblocks waiter", False,
                         f"Got: {error_holder2[0]}"))

    return results


def test_pp_module_map():
    """Test PPModuleMap edge cases without vLLM distributed init."""
    # PPModuleMap imports vllm.distributed.utils.get_pp_indices which
    # requires vLLM distributed init. We test the logic patterns instead.
    results = []

    try:
        from nnsight.modeling.vllm.pp import (
            _FIRST_RANK_MODULES,
            _LAST_RANK_MODULES,
            _LAYER_CONTAINER_NAMES,
        )
        # Verify the module sets contain expected entries for GPT-2
        results.append(("FIRST_RANK has wte", "wte" in _FIRST_RANK_MODULES))
        results.append(("FIRST_RANK has wpe", "wpe" in _FIRST_RANK_MODULES))
        results.append(("LAST_RANK has ln_f", "ln_f" in _LAST_RANK_MODULES))
        results.append(("LAST_RANK has lm_head", "lm_head" in _LAST_RANK_MODULES))
        results.append(("LAST_RANK has logits", "logits" in _LAST_RANK_MODULES))
        results.append(("LAST_RANK has samples", "samples" in _LAST_RANK_MODULES))
        results.append(("LAYER_CONTAINERS has h", "h" in _LAYER_CONTAINER_NAMES))
        results.append(("LAYER_CONTAINERS has layers", "layers" in _LAYER_CONTAINER_NAMES))
    except Exception as e:
        results.append(("PPModuleMap imports", False, str(e)))

    return results


# ---------------------------------------------------------------------------
# Part 1: Write the worker script that does actual profiling in subprocesses
# ---------------------------------------------------------------------------

WORKER_SCRIPT_CONTENT = r'''#!/usr/bin/env python
"""
Worker subprocess for PP profiling. Each invocation creates a VLLM model,
runs a scenario N times, and writes timing results as JSON.
"""

import argparse
import json
import sys
import time
import traceback

import torch


def make_model(pp_size):
    from nnsight.modeling.vllm import VLLM
    kwargs = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.1,
        "dispatch": True,
    }
    if pp_size > 1:
        kwargs["pipeline_parallel_size"] = pp_size
    return VLLM("openai-community/gpt2", **kwargs)


def timed(fn, n_warmup=2, n_runs=10):
    """Run fn n_warmup + n_runs times, return list of n_runs durations in seconds."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


# --- Scenario A: Baseline (no interventions) ---

def scenario_baseline(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            pass
    return {"times": timed(fn, n_runs=n_runs)}


# --- Scenario B: Save logits (last-stage module) ---

def scenario_save_logits(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            logits = model.logits.output.save()
    return {"times": timed(fn, n_runs=n_runs)}


# --- Scenario C: Save early layer (first-stage module) ---

def scenario_save_early(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            h0 = model.transformer.h[0].output[0].save()
    return {"times": timed(fn, n_runs=n_runs)}


# --- Scenario D: Save from both stages ---

def scenario_save_both(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            h0 = model.transformer.h[0].output[0].save()
            h11 = model.transformer.h[11].output[0].save()
    return {"times": timed(fn, n_runs=n_runs)}


# --- Scenario E: Cross-stage write (no materialization expected) ---

def scenario_cross_write(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            model.transformer.h[8].output[0][:] = model.transformer.h[2].output[0]
    return {"times": timed(fn, n_runs=n_runs)}


# --- Scenario F: Cross-stage read (forces materialization) ---

def scenario_cross_read(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            h = model.transformer.h[2].output[0]
            result = (h * 2).save()
    return {"times": timed(fn, n_runs=n_runs)}


# --- Scenario G: Multi-token generation ---

def scenario_multigen(model, prompt, n_runs, max_tokens=3):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=max_tokens) as tracer:
            logit_list = list().save()
            for step in tracer.iter[:]:
                logit_list.append(model.logits.output)
    return {"times": timed(fn, n_warmup=1, n_runs=n_runs)}


# --- Scenario J: Long-distance dependency ---

def scenario_long_distance(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            wte_out = model.transformer.wte.output
            # Use the embedding to modify the last layer output
            model.transformer.h[11].output[0][:] = wte_out
            logits = model.logits.output.save()
    return {"times": timed(fn, n_runs=n_runs)}


# --- Scenario K: Multiple independent traces ---

def scenario_multi_trace(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            logits1 = model.logits.output.save()
        with model.trace("Hello world", temperature=0.0, top_p=1):
            logits2 = model.logits.output.save()
    return {"times": timed(fn, n_runs=n_runs)}


# --- Scenario L: Save all 12 layers ---

def scenario_save_all_layers(model, prompt, n_runs):
    def fn():
        with model.trace(prompt, temperature=0.0, top_p=1):
            saved = []
            for i in range(12):
                saved.append(model.transformer.h[i].output[0].save())
    return {"times": timed(fn, n_runs=n_runs)}


SCENARIOS = {
    "baseline": scenario_baseline,
    "save_logits": scenario_save_logits,
    "save_early": scenario_save_early,
    "save_both": scenario_save_both,
    "cross_write": scenario_cross_write,
    "cross_read": scenario_cross_read,
    "multigen": scenario_multigen,
    "long_distance": scenario_long_distance,
    "multi_trace": scenario_multi_trace,
    "save_all_layers": scenario_save_all_layers,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=list(SCENARIOS.keys()))
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--prompt", type=str, default="The Eiffel Tower is located in the city of")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=3)
    args = parser.parse_args()

    try:
        model = make_model(args.pp)

        scenario_fn = SCENARIOS[args.scenario]
        if args.scenario == "multigen":
            result = scenario_fn(model, args.prompt, args.n_runs, max_tokens=args.max_tokens)
        else:
            result = scenario_fn(model, args.prompt, args.n_runs)

        result["status"] = "ok"

    except Exception as e:
        result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    with open(args.output, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
'''


def write_worker_script():
    """Write the worker script to disk."""
    with open(PROFILE_WORKER, "w") as f:
        f.write(WORKER_SCRIPT_CONTENT)
    os.chmod(PROFILE_WORKER, 0o755)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    free_gpus = check_gpus()
    gpu_pp1 = str(free_gpus[0])
    gpu_pp2 = f"{free_gpus[1]},{free_gpus[2]}"

    print("=" * 70)
    print("  NNsight vLLM PP Profiling & Corner Case Tests")
    print("=" * 70)
    print(f"  GPUs available: {torch.cuda.device_count()}")
    print(f"  Free GPUs: {free_gpus}")
    print(f"  PP=1 GPU: {gpu_pp1}")
    print(f"  PP=2 GPUs: {gpu_pp2}")
    print()

    # Write the worker script
    write_worker_script()
    print(f"  Worker script: {PROFILE_WORKER}")

    # ===================================================================
    # Part 2: Corner case tests (local, no subprocess needed)
    # ===================================================================

    print_header("Part 2: Corner Case Tests (local)")

    # --- Test H: No-materialization operations ---
    print()
    print("  [H] LazyRemoteTensor: operations that should NOT materialize")
    print("  " + "-" * 60)
    try:
        h_results = test_lazy_no_materialization()
        for item in h_results:
            label = item[0]
            passed = item[1]
            detail = item[2] if len(item) > 2 else ""
            print_corner_case(label, passed, detail)
    except Exception as e:
        print(f"  [FAIL] Test H crashed: {e}")
        traceback.print_exc()

    # --- Test I: Materialization operations ---
    print()
    print("  [I] LazyRemoteTensor: operations that SHOULD materialize")
    print("  " + "-" * 60)
    try:
        i_results = test_lazy_materialization()
        for item in i_results:
            label = item[0]
            passed = item[1]
            detail = item[2] if len(item) > 2 else ""
            print_corner_case(label, passed, detail)
    except Exception as e:
        print(f"  [FAIL] Test I crashed: {e}")
        traceback.print_exc()

    # --- PPListener corner cases ---
    print()
    print("  [PPListener] Corner cases")
    print("  " + "-" * 60)
    try:
        listener_results = test_pp_listener_corner_cases()
        for item in listener_results:
            label = item[0]
            passed = item[1]
            detail = item[2] if len(item) > 2 else ""
            print_corner_case(label, passed, detail)
    except Exception as e:
        print(f"  [FAIL] PPListener tests crashed: {e}")
        traceback.print_exc()

    # --- PPModuleMap constants ---
    print()
    print("  [PPModuleMap] Module classification constants")
    print("  " + "-" * 60)
    try:
        map_results = test_pp_module_map()
        for item in map_results:
            label = item[0]
            passed = item[1]
            detail = item[2] if len(item) > 2 else ""
            print_corner_case(label, passed, detail)
    except Exception as e:
        print(f"  [FAIL] PPModuleMap tests crashed: {e}")
        traceback.print_exc()

    # ===================================================================
    # Part 1: Overhead Profiling (subprocess-based)
    # ===================================================================

    print_header("Part 1: Overhead Profiling (PP=1 vs PP=2, 10 runs each)")
    print()
    print("  NOTE: PP=2 tests are expected to fail if mediator deserialization")
    print("  or _pp_aware_load is not yet fully implemented. Errors are caught")
    print("  and reported inline.")
    print()

    N_RUNS = 10
    PROMPT = "The Eiffel Tower is located in the city of"

    scenarios = [
        ("A", "baseline",       "Baseline: trace, no interventions"),
        ("B", "save_logits",    "Save logits (last-stage module)"),
        ("C", "save_early",     "Save h[0] (first-stage module)"),
        ("D", "save_both",      "Save h[0] + h[11] (cross-stage saves)"),
        ("E", "cross_write",    "Cross-stage write (h[2] -> h[8], no materialization)"),
        ("F", "cross_read",     "Cross-stage read (h[2] * 2, forces materialization)"),
        ("G", "multigen",       "Multi-token generation (3 tokens)"),
        ("J", "long_distance",  "Long-distance: wte -> h[11]"),
        ("K", "multi_trace",    "Two independent traces back-to-back"),
        ("L", "save_all_layers","Save all 12 layers"),
    ]

    print(f"  {'Scenario':<50s}  {'PP=1 Mean':>12s} {'PP=1 Std':>10s}  |  "
          f"{'PP=2 Mean':>12s} {'PP=2 Std':>10s}  |  {'Overhead':>10s}")
    print("  " + "-" * 120)

    for tag, scenario, label in scenarios:
        full_label = f"[{tag}] {label}"

        extra_args = ["--n_runs", str(N_RUNS)]
        if scenario == "multigen":
            extra_args.extend(["--max_tokens", "3"])

        # Run PP=1
        pp1_data = _run_worker(gpu_pp1, scenario, pp=1, prompt=PROMPT, extra_args=extra_args)

        # Run PP=2
        pp2_data = _run_worker(gpu_pp2, scenario, pp=2, prompt=PROMPT, extra_args=extra_args)

        print_timing_row(full_label, pp1_data, pp2_data)

    # ===================================================================
    # Detailed error reporting for failed scenarios
    # ===================================================================

    print_header("Detailed Error Reports (if any)")

    any_errors = False
    for tag, scenario, label in scenarios:
        extra_args = ["--n_runs", str(N_RUNS)]
        if scenario == "multigen":
            extra_args.extend(["--max_tokens", "3"])

        for pp_size, gpu_str in [(1, gpu_pp1), (2, gpu_pp2)]:
            data = _run_worker(gpu_str, scenario, pp=pp_size, prompt=PROMPT, extra_args=extra_args)
            if data.get("status") != "ok":
                any_errors = True
                print(f"\n  [{tag}] PP={pp_size} {label}")
                print(f"  Error: {data.get('error', 'unknown')[:500]}")
                if "stderr" in data:
                    stderr_lines = data["stderr"].strip().split("\n")
                    for line in stderr_lines[-20:]:
                        print(f"    {line}")
                if "traceback" in data:
                    tb_lines = data["traceback"].strip().split("\n")
                    for line in tb_lines[-15:]:
                        print(f"    {line}")

    if not any_errors:
        print("  No errors.")

    # ===================================================================
    # Summary
    # ===================================================================

    print_header("Summary")

    # Count Part 2 results
    all_corner_cases = []
    try:
        all_corner_cases.extend(test_lazy_no_materialization())
    except Exception:
        pass
    try:
        all_corner_cases.extend(test_lazy_materialization())
    except Exception:
        pass
    try:
        all_corner_cases.extend(test_pp_listener_corner_cases())
    except Exception:
        pass
    try:
        all_corner_cases.extend(test_pp_module_map())
    except Exception:
        pass

    passed = sum(1 for r in all_corner_cases if r[1])
    failed = sum(1 for r in all_corner_cases if not r[1])
    total = len(all_corner_cases)

    print(f"  Corner case tests: {passed}/{total} passed, {failed} failed")
    print(f"  Profiling scenarios: {len(scenarios)} (see table above)")
    print()

    # Cleanup worker script
    try:
        os.unlink(PROFILE_WORKER)
    except OSError:
        pass


if __name__ == "__main__":
    main()
