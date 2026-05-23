"""Stress test: concurrent requests to nnsight-vllm-serve with TP/PP.

Phases:
  A  -- 9 concurrent distinct-prompt / distinct-layer traces; every
        request must receive its OWN saves, not a neighbor's. Catches
        cross-trace entanglement (Bug A) and pull-protocol buffer
        collisions (Bug B). Acceptance: 9/9 OK.

  C2 -- 16 concurrent traces, each on its own prompt, saving the last
        hidden state of a distinct layer. Compares against a
        sequential ground-truth pass to detect leakage.
        Acceptance: >= 15/16 match.

  D  -- Throughput: 50 requests through 8 concurrent workers, measure
        end-to-end req/s. Acceptance: >= 7 req/s.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python tests/stress_pp_tp_serve.py \
      --configs tp2pp2 --phases A,C,D

All client requests go through ``LanguageModel``/``VLLM`` with
``serve=<url>``; the script only needs ``httpx`` and an installed
``nnsight`` locally.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Config presets
# --------------------------------------------------------------------------- #

@dataclass
class ServerConfig:
    name: str
    tp: int
    pp: int
    port: int
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    gpu_mem_util: float = 0.5
    max_model_len: int = 4096


CONFIGS: dict[str, ServerConfig] = {
    "tp2pp2": ServerConfig(name="tp2pp2", tp=2, pp=2, port=6704),
    "tp1pp2": ServerConfig(name="tp1pp2", tp=1, pp=2, port=6701),
    "tp2pp1": ServerConfig(name="tp2pp1", tp=2, pp=1, port=6702),
    "tp1pp1": ServerConfig(name="tp1pp1", tp=1, pp=1, port=6703),
}


# --------------------------------------------------------------------------- #
# Prompts: distinct enough that their last-token hidden states diverge.
# --------------------------------------------------------------------------- #

PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "The largest ocean on Earth is called the",
    "The chemical symbol for gold is",
    "The author of Hamlet was William",
    "The speed of light is approximately 299,792",
    "The capital of Japan is the city of",
    "The first president of the United States was George",
    "The deepest point in the ocean is the Mariana",
    "The currency of the United Kingdom is the British",
    "The longest river in the world is the",
    "The planet closest to the Sun is named",
    "The tallest mountain in the world is called Mount",
    "The fastest land animal is the",
    "The fundamental unit of heredity is the",
    "The number of colors in a rainbow is commonly said to be",
    "The process by which plants make food is called",
    "The inventor of the lightbulb was Thomas",
    "The oldest written language in the world is",
]


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #

def start_server(cfg: ServerConfig, devices: str, log_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    # Keep vLLM / nnsight from spraying noise into our parsed stdout.
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    cmd = [
        sys.executable, "-m", "nnsight.modeling.vllm.serve.cli",
        cfg.model,
        "--host", "127.0.0.1",
        "--port", str(cfg.port),
        "--tensor-parallel-size", str(cfg.tp),
        "--pipeline-parallel-size", str(cfg.pp),
        "--gpu-memory-utilization", str(cfg.gpu_mem_util),
        "--max-model-len", str(cfg.max_model_len),
    ]
    print(f"[server] launching: {' '.join(cmd[-10:])}", flush=True)
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    finally:
        fp = getattr(proc, "_log_fp", None)
        if fp is not None:
            fp.close()


def wait_for_server(url: str, timeout: float = 600.0) -> bool:
    import httpx

    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception as e:
            last_err = e
        time.sleep(2)
    print(f"[server] never became healthy: {last_err}", flush=True)
    return False


# --------------------------------------------------------------------------- #
# Trace helpers
# --------------------------------------------------------------------------- #

def make_model(hf_name: str):
    """Build a meta-device VLLM wrapper for client-side trace compilation."""
    from nnsight.modeling.vllm import VLLM

    # No dispatch / no GPU: client only needs module structure for envoy paths.
    return VLLM(hf_name)


def trace_layer_hidden(
    model,
    prompt: str,
    layer: int,
    server_url: str,
    max_tokens: int = 1,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one trace saving last-token hidden at ``model.model.layers[layer]``."""
    import torch

    t0 = time.perf_counter()
    try:
        with model.trace(
            prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            serve=server_url,
        ):
            hs = model.model.layers[layer].output[0].save()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if not isinstance(hs, torch.Tensor):
            return {"ok": False, "err": f"non-tensor: {type(hs).__name__}", "ms": elapsed_ms}
        # vLLM returns token-flat [total_tokens, hidden] during prefill,
        # not HF-style [batch, seq, hidden].  Handle both shapes; reduce
        # to the last-token hidden vector so comparisons are cheap.
        if hs.ndim == 3:
            last = hs[:, -1, :].detach().cpu()
        elif hs.ndim == 2:
            last = hs[-1:, :].detach().cpu()
        else:
            return {"ok": False, "err": f"unexpected ndim={hs.ndim}", "ms": elapsed_ms}
        return {
            "ok": True,
            "shape": tuple(last.shape),
            "tensor": last,
            "ms": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        msg = str(e)
        return {"ok": False, "err": msg[:240], "ms": elapsed_ms}


# --------------------------------------------------------------------------- #
# Ground truth: sequential single-request baseline for isolation checks.
# --------------------------------------------------------------------------- #

def collect_ground_truth(
    model,
    server_url: str,
    cases: list[tuple[int, str, int]],  # (idx, prompt, layer)
) -> dict[int, "torch.Tensor"]:
    """One request at a time — no chance of cross-trace contamination."""
    gt: dict[int, Any] = {}
    for idx, prompt, layer in cases:
        res = trace_layer_hidden(model, prompt, layer, server_url)
        if not res["ok"]:
            print(f"[gt] case {idx} FAILED: {res['err']}", flush=True)
            continue
        gt[idx] = res["tensor"]
    return gt


def tensor_match(a, b, leak_threshold: float = 2.0) -> tuple[bool, float]:
    """Return (match, max_abs_diff).

    The ground-truth pass runs requests sequentially (batch=1); the
    stress pass runs them concurrently (larger batches).  Flash-attn
    kernels produce bit-slightly different results at different batch
    sizes, so ``max|Δ|`` of a few tenths is expected kernel noise, NOT
    a cross-trace leak.  A true leak (request i receiving request j's
    hidden state) produces diffs orders of magnitude larger -- prior
    observations in this codebase ranged 10^3 to 10^4 -- since the
    two hidden vectors come from entirely different prompts.

    ``leak_threshold`` sits well above observed batch noise
    (~0.01-0.5) and well below cross-trace leak magnitudes.
    """
    import torch
    if a is None or b is None:
        return False, float("inf")
    if a.shape != b.shape:
        return False, float("inf")
    diff = (a.float() - b.float()).abs().max().item()
    return diff < leak_threshold, diff


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

@dataclass
class PhaseResult:
    name: str
    ok: int
    total: int
    wall_s: float
    req_per_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def passed(self, *, min_ok: int | None = None, min_req_s: float | None = None) -> bool:
        if min_ok is not None and self.ok < min_ok:
            return False
        if min_req_s is not None and self.req_per_s < min_req_s:
            return False
        return True


def _choose_layers(n: int, num_hidden_layers: int) -> list[int]:
    """Pick ``n`` distinct layers, spaced across the stack."""
    if n <= 0:
        return []
    if n == 1:
        return [num_hidden_layers // 2]
    step = max(1, num_hidden_layers // n)
    out = [min(num_hidden_layers - 1, i * step) for i in range(n)]
    # de-dup while preserving order
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    # fill up if de-dup shrank the list
    i = 0
    while len(dedup) < n and i < num_hidden_layers:
        if i not in seen:
            dedup.append(i)
            seen.add(i)
        i += 1
    return dedup[:n]


def phase_A(model, server_url: str, num_hidden_layers: int) -> PhaseResult:
    """9 concurrent distinct traces, verify each gets its own saves."""
    n = 9
    layers = _choose_layers(n, num_hidden_layers)
    cases = [(i, PROMPTS[i], layers[i]) for i in range(n)]

    print(f"[A] collecting ground truth ({n} sequential requests)", flush=True)
    gt = collect_ground_truth(model, server_url, cases)
    if len(gt) < n:
        print(f"[A] GROUND-TRUTH incomplete: {len(gt)}/{n} -- aborting", flush=True)
        return PhaseResult(name="A", ok=0, total=n, wall_s=0.0)

    print(f"[A] firing {n} concurrent traces", flush=True)
    results: dict[int, dict] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {
            pool.submit(trace_layer_hidden, model, prompt, layer, server_url): idx
            for (idx, prompt, layer) in cases
        }
        for f in as_completed(futs):
            idx = futs[f]
            results[idx] = f.result()
    wall = time.perf_counter() - t0

    ok = 0
    mismatches = []
    for idx, _, _ in cases:
        r = results.get(idx)
        if r is None or not r["ok"]:
            mismatches.append((idx, f"error: {r['err'] if r else 'missing'}"))
            continue
        match, diff = tensor_match(r["tensor"], gt[idx])
        if match:
            ok += 1
        else:
            mismatches.append((idx, f"max|Δ|={diff:.4g}"))

    for idx, why in mismatches[:5]:
        print(f"[A] case {idx} MISMATCH: {why}", flush=True)
    print(f"[A] {ok}/{n} OK, wall={wall:.2f}s", flush=True)
    return PhaseResult(name="A", ok=ok, total=n, wall_s=wall,
                       req_per_s=n / wall if wall > 0 else 0.0)


def phase_C2(model, server_url: str, num_hidden_layers: int) -> PhaseResult:
    """16 concurrent isolation -- the canonical Bug B failure case."""
    n = 16
    layers = _choose_layers(n, num_hidden_layers)
    # Reuse the prompt pool with wrap-around; distinct layers keep each
    # request's expected hidden state unique even when prompts repeat.
    cases = [(i, PROMPTS[i % len(PROMPTS)], layers[i]) for i in range(n)]

    print(f"[C2] collecting ground truth ({n} sequential requests)", flush=True)
    gt = collect_ground_truth(model, server_url, cases)
    if len(gt) < n:
        print(f"[C2] GROUND-TRUTH incomplete: {len(gt)}/{n} -- aborting", flush=True)
        return PhaseResult(name="C2", ok=0, total=n, wall_s=0.0)

    print(f"[C2] firing {n} concurrent traces", flush=True)
    results: dict[int, dict] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {
            pool.submit(trace_layer_hidden, model, prompt, layer, server_url): idx
            for (idx, prompt, layer) in cases
        }
        for f in as_completed(futs):
            idx = futs[f]
            results[idx] = f.result()
    wall = time.perf_counter() - t0

    ok = 0
    mismatches = []
    max_diff = 0.0
    for idx, _, _ in cases:
        r = results.get(idx)
        if r is None or not r["ok"]:
            mismatches.append((idx, f"error: {r['err'] if r else 'missing'}"))
            continue
        match, diff = tensor_match(r["tensor"], gt[idx])
        max_diff = max(max_diff, diff if diff != float("inf") else max_diff)
        if match:
            ok += 1
        else:
            mismatches.append((idx, f"max|Δ|={diff:.4g}"))

    for idx, why in mismatches[:8]:
        print(f"[C2] case {idx} MISMATCH: {why}", flush=True)
    print(f"[C2] {ok}/{n} OK (max|Δ|={max_diff:.4g}), wall={wall:.2f}s", flush=True)
    return PhaseResult(
        name="C2", ok=ok, total=n, wall_s=wall,
        req_per_s=n / wall if wall > 0 else 0.0,
        extra={"max_abs_diff": max_diff},
    )


def phase_D(model, server_url: str, num_hidden_layers: int) -> PhaseResult:
    """Throughput: 50 requests through 8 concurrent workers."""
    total = 50
    workers = 8
    layer_mid = num_hidden_layers // 2
    # Simplest possible trace -- one save per request -- so throughput
    # measures serving/dispatch cost, not client compute.
    tasks = [(i, PROMPTS[i % len(PROMPTS)], layer_mid) for i in range(total)]

    print(f"[D] firing {total} requests through {workers} workers", flush=True)
    results: dict[int, dict] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(trace_layer_hidden, model, prompt, layer, server_url): idx
            for (idx, prompt, layer) in tasks
        }
        for f in as_completed(futs):
            idx = futs[f]
            results[idx] = f.result()
    wall = time.perf_counter() - t0

    ok = sum(1 for r in results.values() if r["ok"])
    rps = total / wall if wall > 0 else 0.0
    latencies = sorted(r["ms"] for r in results.values() if r["ok"])
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    print(f"[D] {ok}/{total} OK, wall={wall:.2f}s ({rps:.2f} req/s) "
          f"p50={p50:.0f}ms p95={p95:.0f}ms", flush=True)
    return PhaseResult(
        name="D", ok=ok, total=total, wall_s=wall, req_per_s=rps,
        extra={"p50_ms": p50, "p95_ms": p95},
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

ACCEPTANCE = {
    "A":  dict(min_ok=9),
    "C2": dict(min_ok=15),
    # Phase D: require BOTH acceptable throughput AND most requests
    # actually succeeding — a fast string of HTTP 500s is not a pass.
    "D":  dict(min_ok=45, min_req_s=7.0),
}


def run_config(
    cfg: ServerConfig,
    phases: list[str],
    devices: str,
    logs_dir: str,
) -> dict[str, PhaseResult]:
    server_url = f"http://127.0.0.1:{cfg.port}"
    log_path = os.path.join(logs_dir, f"server_{cfg.name}.log")
    proc = start_server(cfg, devices, log_path)
    results: dict[str, PhaseResult] = {}
    try:
        print(f"[{cfg.name}] waiting for /health (this may take a minute)", flush=True)
        if not wait_for_server(server_url):
            print(f"[{cfg.name}] server did not become healthy -- see {log_path}", flush=True)
            return results

        model = make_model(cfg.model)
        # Ping the layers attribute once so downstream phases don't race
        # on one-time source-compilation overhead.
        _ = len(model.model.layers)  # type: ignore[attr-defined]
        num_layers = len(model.model.layers)

        phase_fns: dict[str, Callable] = {
            "A":  phase_A,
            "C":  phase_C2,   # the summary uses "C" interchangeably with "C2"
            "C2": phase_C2,
            "D":  phase_D,
        }

        for ph in phases:
            fn = phase_fns.get(ph.upper())
            if fn is None:
                print(f"[{cfg.name}] unknown phase '{ph}' -- skipping", flush=True)
                continue
            key = ph.upper() if ph.upper() in ACCEPTANCE else "C2" if ph.upper() == "C" else ph.upper()
            pr = fn(model, server_url, num_layers)
            results[key] = pr
    finally:
        stop_server(proc)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="tp2pp2",
                        help="comma-separated config names (see CONFIGS)")
    parser.add_argument("--phases", default="A,C,D",
                        help="comma-separated phase names (A, C or C2, D)")
    parser.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "4,5,6,7"),
                        help="CUDA_VISIBLE_DEVICES for the server process")
    parser.add_argument("--logs-dir", default="/tmp/stress_pp_tp")
    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    overall: dict[str, dict[str, PhaseResult]] = {}
    for name in config_names:
        cfg = CONFIGS.get(name)
        if cfg is None:
            print(f"[main] unknown config '{name}' -- skipping", flush=True)
            continue
        print(f"\n===== config {cfg.name} (tp={cfg.tp}, pp={cfg.pp}) =====", flush=True)
        overall[name] = run_config(cfg, phases, args.devices, args.logs_dir)

    # Summary & acceptance
    print("\n================ SUMMARY ================", flush=True)
    all_pass = True
    for name, phase_results in overall.items():
        print(f"\n-- {name} --", flush=True)
        for key, pr in phase_results.items():
            crit = ACCEPTANCE.get(key if key in ACCEPTANCE else "C2", {})
            passed = pr.passed(**crit)
            tag = "PASS" if passed else "FAIL"
            line = f"  [{tag}] Phase {key}: {pr.ok}/{pr.total} OK, " \
                   f"{pr.req_per_s:.2f} req/s, wall={pr.wall_s:.2f}s"
            if pr.extra:
                line += f" extra={pr.extra}"
            print(line, flush=True)
            if not passed:
                all_pass = False

    # Persist machine-readable summary too
    summary_path = os.path.join(args.logs_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                cfg_name: {
                    ph: {
                        "ok": pr.ok,
                        "total": pr.total,
                        "wall_s": pr.wall_s,
                        "req_per_s": pr.req_per_s,
                        "extra": pr.extra,
                    }
                    for ph, pr in phase_results.items()
                }
                for cfg_name, phase_results in overall.items()
            },
            f,
            indent=2,
        )
    print(f"\nsummary written to {summary_path}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
