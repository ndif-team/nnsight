"""
Batching performance benchmark for NNsight.

Validates claims from the inference engine analysis:
- Hook overhead baseline (Scenario 1)
- Mediator scaling O(modules x mediators) (Scenario 2)
- Sequential vs batched throughput (Scenario 3)
- Head-of-line blocking from heavy interventions (Scenario 4)
- CUDA sync impact on read vs write (Scenario 5)
- vLLM backend comparison (Scenario 6)

Run with:
    python tests/performance/benchmark_batching.py                   # Full run (Qwen2.5-7B)
    python tests/performance/benchmark_batching.py --smoke           # Quick (GPT-2)
    python tests/performance/benchmark_batching.py --backend vllm    # vLLM backend
    python tests/performance/benchmark_batching.py --scenarios 1,3   # Selected scenarios
    python tests/performance/benchmark_batching.py --plot            # Generate plots
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field

# Prevent tests/performance/profile/ from shadowing Python's profile module.
# cProfile (used by IPython during nnsight import) needs the real profile module.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "openai-community/gpt2": {
        "layers_attr": ["transformer", "h"],  # model.transformer.h[i]
        "num_layers": 12,
        "hidden_dim": 768,
        "head_module": "lm_head",
        "dtype": torch.float32,
    },
    "Qwen/Qwen2.5-7B": {
        "layers_attr": ["model", "layers"],  # model.model.layers[i]
        "num_layers": 28,
        "hidden_dim": 3584,
        "head_module": "lm_head",
        "dtype": torch.float16,
    },
}

PROMPTS = [f"The answer to question number {i} is that" for i in range(32)]

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
SMOKE_MODEL = "openai-community/gpt2"

ALL_SCENARIOS = [1, 2, 3, 4, 5, 6]

# ---------------------------------------------------------------------------
# Timing utilities (adapted from profiler_utils.py for self-containment)
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    name: str
    times_ms: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    @property
    def std(self) -> float:
        if len(self.times_ms) < 2:
            return 0.0
        m = self.mean
        return (sum((t - m) ** 2 for t in self.times_ms) / (len(self.times_ms) - 1)) ** 0.5

    @property
    def min(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    def as_dict(self) -> dict:
        return {"mean_ms": round(self.mean, 3), "std_ms": round(self.std, 3),
                "min_ms": round(self.min, 3), "max_ms": round(self.max, 3)}


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def force_gc():
    for _ in range(3):
        gc.collect()


def time_fn(fn: Callable, *, warmup: int = 3, runs: int = 10, name: str = "") -> TimingResult:
    """Time *fn* over *runs* iterations after *warmup* warm-ups."""
    for _ in range(warmup):
        fn()
        force_gc()

    result = TimingResult(name=name)
    for _ in range(runs):
        sync_cuda()
        t0 = time.perf_counter()
        fn()
        sync_cuda()
        result.times_ms.append((time.perf_counter() - t0) * 1000)
        force_gc()
    return result


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def get_model_config(model_name: str) -> dict:
    """Return config for a known model or build a best-effort default."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    # Fallback: try to load and inspect
    raise ValueError(
        f"Unknown model {model_name!r}. Add it to MODEL_CONFIGS or use a known model."
    )


def get_layers_module(model, cfg: dict):
    """Resolve the layers container (e.g. model.transformer.h or model.model.layers)."""
    obj = model
    for attr in cfg["layers_attr"]:
        obj = getattr(obj, attr)
    return obj


def get_layer(model, cfg: dict, idx: int):
    """Return the envoy / module for layer *idx*."""
    return get_layers_module(model, cfg)[idx]


def get_head(model, cfg: dict):
    """Return the lm_head envoy."""
    return getattr(model, cfg["head_module"])


def load_transformers(model_name: str, cfg: dict):
    """Load a LanguageModel for the transformers backend."""
    from nnsight import LanguageModel

    kwargs = {"device_map": "cuda:0", "dispatch": True}
    if cfg["dtype"] == torch.float16:
        kwargs["torch_dtype"] = torch.float16
    model = LanguageModel(model_name, **kwargs)
    return model


def load_vllm(model_name: str, cfg: dict):
    """Load a VLLM model. Returns None if vllm is unavailable."""
    try:
        from nnsight.modeling.vllm import VLLM
    except Exception:
        return None
    kwargs = {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.5, "dispatch": True}
    return VLLM(model_name, **kwargs)


def tokenize_for_raw(model, prompt: str) -> dict:
    """Tokenize a prompt for raw model forward pass."""
    tok = model.tokenizer(prompt, return_tensors="pt", padding=True)
    device = next(model._model.parameters()).device
    return {k: v.to(device) for k, v in tok.items()}


# ---------------------------------------------------------------------------
# Scenario implementations
# ---------------------------------------------------------------------------


def scenario_1_hook_overhead(model, cfg: dict, prompts: List[str],
                             warmup: int, runs: int) -> dict:
    """Scenario 1: Hook overhead baseline.

    Measures raw forward, empty trace, and single-save overhead.
    """
    print("\n--- Scenario 1: Hook Overhead Baseline ---")
    prompt = prompts[0]
    mid = cfg["num_layers"] // 2

    # 1a. Raw forward (no nnsight hooks)
    inputs = tokenize_for_raw(model, prompt)

    @torch.no_grad()
    def raw_forward():
        model._model(**inputs)

    # Use at least 5 warmups for raw forward to cover CUDA JIT / context init
    t_raw = time_fn(raw_forward, warmup=max(warmup, 5), runs=runs, name="raw_forward")

    # 1b. Empty trace (all hooks fire, no intervention)
    def empty_trace():
        with model.trace(prompt):
            pass

    t_empty = time_fn(empty_trace, warmup=max(warmup, 3), runs=runs, name="empty_trace")

    # 1c. Single save
    def single_save():
        with model.trace(prompt):
            get_layer(model, cfg, mid).output[0].save()

    t_single = time_fn(single_save, warmup=max(warmup, 3), runs=runs, name="single_save")

    overhead_empty = t_empty.mean / t_raw.mean if t_raw.mean > 0 else float("inf")
    overhead_single = t_single.mean / t_raw.mean if t_raw.mean > 0 else float("inf")

    print(f"  Raw forward:    {t_raw.mean:8.2f} ms  (std {t_raw.std:.2f})")
    print(f"  Empty trace:    {t_empty.mean:8.2f} ms  (std {t_empty.std:.2f})  "
          f"overhead: {overhead_empty:.2f}x")
    print(f"  Single save:    {t_single.mean:8.2f} ms  (std {t_single.std:.2f})  "
          f"overhead: {overhead_single:.2f}x")

    return {
        "raw": t_raw.as_dict(),
        "empty_trace": t_empty.as_dict(),
        "single_save": t_single.as_dict(),
        "overhead_empty": round(overhead_empty, 3),
        "overhead_single": round(overhead_single, 3),
    }


def scenario_2_mediator_scaling(model, cfg: dict, prompts: List[str],
                                warmup: int, runs: int) -> dict:
    """Scenario 2: Mediator scaling – O(modules x mediators).

    Time M=1,2,4,8,16 mediators each doing .save() on one layer.
    """
    print("\n--- Scenario 2: Mediator Scaling ---")
    num_layers = cfg["num_layers"]
    mediator_counts = [1, 2, 4, 8, 16]
    results = {}

    for M in mediator_counts:
        if M > len(prompts):
            break

        def run(m=M):
            with model.trace() as tracer:
                for i in range(m):
                    with tracer.invoke(prompts[i]):
                        get_layer(model, cfg, i % num_layers).output[0].save()

        t = time_fn(run, warmup=warmup, runs=runs, name=f"M={M}")
        results[str(M)] = t.as_dict()
        print(f"  M={M:>2}: {t.mean:8.2f} ms  (std {t.std:.2f})")

    # Linear regression: time = a*M + b
    ms = sorted(int(k) for k in results)
    times = [results[str(m)]["mean_ms"] for m in ms]
    if len(ms) >= 2:
        import statistics
        x_mean = statistics.mean(ms)
        y_mean = statistics.mean(times)
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(ms, times))
        ss_xx = sum((x - x_mean) ** 2 for x in ms)
        slope = ss_xy / ss_xx if ss_xx else 0
        intercept = y_mean - slope * x_mean
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(ms, times))
        ss_tot = sum((y - y_mean) ** 2 for y in times)
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0
        results["regression"] = {
            "slope_ms_per_mediator": round(slope, 3),
            "intercept_ms": round(intercept, 3),
            "r_squared": round(r_squared, 4),
        }
        print(f"  Linear fit: {slope:.2f} ms/mediator, R²={r_squared:.4f}")

    return results


def scenario_3_sequential_vs_batched(model, cfg: dict, prompts: List[str],
                                     warmup: int, runs: int) -> dict:
    """Scenario 3: Batching overhead – raw vs nnsight.

    For B=1,2,4,8,16,32: compare raw batched forward (no hooks) vs nnsight
    batched (1 trace, B invokes each with .save()).  The claim is that
    mediator overhead grows with B, so nnsight-batched scales worse than raw.
    """
    print("\n--- Scenario 3: Sequential vs Batched (raw baseline) ---")
    mid = cfg["num_layers"] // 2
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = {}

    for B in batch_sizes:
        if B > len(prompts):
            break

        # Raw batched: tokenize B prompts, single model forward, no hooks
        raw_inputs = model.tokenizer(
            prompts[:B], return_tensors="pt", padding=True,
        )
        device = next(model._model.parameters()).device
        raw_inputs = {k: v.to(device) for k, v in raw_inputs.items()}

        @torch.no_grad()
        def raw_batched(inp=raw_inputs):
            model._model(**inp)

        t_raw = time_fn(raw_batched, warmup=max(warmup, 3), runs=runs, name=f"raw_B={B}")

        # NNsight batched: 1 trace, B invokes, each with 1 .save()
        def nnsight_batched(b=B):
            with model.trace() as tracer:
                for p in prompts[:b]:
                    with tracer.invoke(p):
                        get_layer(model, cfg, mid).output[0].save()

        t_nn = time_fn(nnsight_batched, warmup=warmup, runs=runs, name=f"nn_B={B}")

        overhead = t_nn.mean / t_raw.mean if t_raw.mean > 0 else float("inf")
        overhead_abs = t_nn.mean - t_raw.mean
        throughput_raw = B / (t_raw.mean / 1000) if t_raw.mean > 0 else 0
        throughput_nn = B / (t_nn.mean / 1000) if t_nn.mean > 0 else 0

        results[str(B)] = {
            "raw_batched": t_raw.as_dict(),
            "nnsight_batched": t_nn.as_dict(),
            "overhead_ratio": round(overhead, 3),
            "overhead_abs_ms": round(overhead_abs, 2),
            "throughput_raw": round(throughput_raw, 1),
            "throughput_nn": round(throughput_nn, 1),
        }
        print(f"  B={B:>2}:  raw={t_raw.mean:8.2f}ms  nnsight={t_nn.mean:8.2f}ms  "
              f"overhead={overhead:.2f}x (+{overhead_abs:.1f}ms)  "
              f"thr: {throughput_raw:.0f} vs {throughput_nn:.0f} req/s")

    return results


def scenario_4_hol_blocking(model, cfg: dict, prompts: List[str],
                            warmup: int, runs: int) -> dict:
    """Scenario 4: Head-of-line blocking from intervention weight.

    For B=8, test save_only / light / medium / heavy interventions.
    """
    print("\n--- Scenario 4: Head-of-Line Blocking ---")
    B = 8
    if B > len(prompts):
        B = len(prompts)
    mid = cfg["num_layers"] // 2
    hidden_dim = cfg["hidden_dim"]
    dtype = cfg["dtype"]

    device = "cuda"

    # Pre-allocate projection matrices
    W_med = torch.randn(hidden_dim, hidden_dim, device=device, dtype=dtype)
    expansion = 4 * hidden_dim
    W_enc = torch.randn(hidden_dim, expansion, device=device, dtype=dtype)
    W_dec = torch.randn(expansion, hidden_dim, device=device, dtype=dtype)

    def make_intervention(weight: str):
        """Return (sequential_fn, batched_fn) for a given intervention weight."""

        if weight == "save_only":
            def _invoke_body(m, c):
                get_layer(m, c, mid).output[0].save()
        elif weight == "light":
            def _invoke_body(m, c):
                h = get_layer(m, c, mid).output[0]
                _ = h.mean()
                h.save()
        elif weight == "medium":
            def _invoke_body(m, c):
                h = get_layer(m, c, mid).output[0]
                projected = h @ W_med
                get_layer(m, c, mid).output[0][:] = projected
                get_head(m, c).output.save()
        elif weight == "heavy":
            def _invoke_body(m, c):
                h = get_layer(m, c, mid).output[0]
                encoded = torch.relu(h @ W_enc)
                decoded = encoded @ W_dec
                get_layer(m, c, mid).output[0][:] = decoded
                get_head(m, c).output.save()
        else:
            raise ValueError(weight)

        def seq(b=B, body=_invoke_body):
            for p in prompts[:b]:
                with model.trace(p):
                    body(model, cfg)

        def bat(b=B, body=_invoke_body):
            with model.trace() as tracer:
                for p in prompts[:b]:
                    with tracer.invoke(p):
                        body(model, cfg)

        return seq, bat

    weights = ["save_only", "light", "medium", "heavy"]
    results = {}
    for w in weights:
        seq_fn, bat_fn = make_intervention(w)

        t_seq = time_fn(seq_fn, warmup=warmup, runs=runs, name=f"seq_{w}")
        t_bat = time_fn(bat_fn, warmup=warmup, runs=runs, name=f"bat_{w}")

        speedup = t_seq.mean / t_bat.mean if t_bat.mean > 0 else float("inf")
        results[w] = {
            "sequential": t_seq.as_dict(),
            "batched": t_bat.as_dict(),
            "speedup": round(speedup, 3),
        }
        print(f"  {w:<12}: seq={t_seq.mean:8.2f}ms  bat={t_bat.mean:8.2f}ms  "
              f"speedup={speedup:.2f}x")

    return results


def scenario_5_cuda_sync(model, cfg: dict, prompts: List[str],
                         warmup: int, runs: int) -> dict:
    """Scenario 5: CUDA sync impact.

    Three variants: save_only, value_read (.mean()), in_place_modify.
    """
    print("\n--- Scenario 5: CUDA Sync Impact ---")
    mid = cfg["num_layers"] // 2
    prompt = prompts[0]

    # save_only
    def fn_save():
        with model.trace(prompt):
            get_layer(model, cfg, mid).output[0].save()

    t_save = time_fn(fn_save, warmup=warmup, runs=runs, name="save_only")

    # value_read – forces sync via .mean()
    def fn_read():
        with model.trace(prompt):
            h = get_layer(model, cfg, mid).output[0]
            _ = h.mean()
            h.save()

    t_read = time_fn(fn_read, warmup=warmup, runs=runs, name="value_read")

    # in_place_modify – async kernel, no sync
    def fn_inplace():
        with model.trace(prompt):
            get_layer(model, cfg, mid).output[0][:] = 0
            get_layer(model, cfg, mid).output[0].save()

    t_inplace = time_fn(fn_inplace, warmup=warmup, runs=runs, name="in_place_modify")

    print(f"  save_only:       {t_save.mean:8.2f} ms  (std {t_save.std:.2f})")
    print(f"  value_read:      {t_read.mean:8.2f} ms  (std {t_read.std:.2f})")
    print(f"  in_place_modify: {t_inplace.mean:8.2f} ms  (std {t_inplace.std:.2f})")

    delta_read = (t_read.mean - t_save.mean) / t_save.mean * 100 if t_save.mean > 0 else 0
    delta_inplace = (t_inplace.mean - t_save.mean) / t_save.mean * 100 if t_save.mean > 0 else 0

    print(f"  value_read overhead:      {delta_read:+.1f}%")
    print(f"  in_place_modify overhead: {delta_inplace:+.1f}%")

    return {
        "save_only": t_save.as_dict(),
        "value_read": t_read.as_dict(),
        "in_place_modify": t_inplace.as_dict(),
        "delta_read_pct": round(delta_read, 2),
        "delta_inplace_pct": round(delta_inplace, 2),
    }


def scenario_6_vllm(model_name: str, cfg: dict, prompts: List[str],
                     warmup: int, runs: int) -> Optional[dict]:
    """Scenario 6: vLLM backend comparison (scenarios 1-4 adapted)."""
    print("\n--- Scenario 6: vLLM Backend ---")

    vllm_model = load_vllm(model_name, cfg)
    if vllm_model is None:
        print("  SKIPPED: vLLM not available")
        return None

    mid = cfg["num_layers"] // 2
    hidden_dim = cfg["hidden_dim"]
    dtype = cfg["dtype"]
    device = "cuda"
    results = {}

    # --- 6a: Hook overhead ---
    print("  [6a] Hook overhead")
    prompt = prompts[0]

    def empty_trace():
        with vllm_model.trace(prompt, temperature=0.0, top_p=1):
            pass

    t_empty = time_fn(empty_trace, warmup=warmup, runs=runs, name="vllm_empty")

    def single_save():
        with vllm_model.trace(prompt, temperature=0.0, top_p=1):
            get_layer(vllm_model, cfg, mid).output[0].save()

    t_single = time_fn(single_save, warmup=warmup, runs=runs, name="vllm_single")

    overhead = t_single.mean / t_empty.mean if t_empty.mean > 0 else float("inf")
    results["hook_overhead"] = {
        "empty_trace": t_empty.as_dict(),
        "single_save": t_single.as_dict(),
        "overhead_ratio": round(overhead, 3),
    }
    print(f"    Empty: {t_empty.mean:.2f}ms  Single: {t_single.mean:.2f}ms  "
          f"overhead={overhead:.2f}x")

    # --- 6b: Mediator scaling ---
    print("  [6b] Mediator scaling")
    num_layers = cfg["num_layers"]
    mediator_results = {}
    for M in [1, 2, 4, 8]:
        if M > len(prompts):
            break

        def run(m=M):
            with vllm_model.trace(temperature=0.0, top_p=1) as tracer:
                for i in range(m):
                    with tracer.invoke(prompts[i]):
                        get_layer(vllm_model, cfg, i % num_layers).output[0].save()

        t = time_fn(run, warmup=warmup, runs=runs, name=f"vllm_M={M}")
        mediator_results[str(M)] = t.as_dict()
        print(f"    M={M}: {t.mean:.2f}ms")

    results["mediator_scaling"] = mediator_results

    # --- 6c: Sequential vs batched ---
    print("  [6c] Sequential vs batched")
    batch_results = {}
    for B in [1, 2, 4, 8]:
        if B > len(prompts):
            break

        def sequential(b=B):
            for p in prompts[:b]:
                with vllm_model.trace(p, temperature=0.0, top_p=1):
                    get_layer(vllm_model, cfg, mid).output[0].save()

        def batched(b=B):
            with vllm_model.trace(temperature=0.0, top_p=1) as tracer:
                for p in prompts[:b]:
                    with tracer.invoke(p):
                        get_layer(vllm_model, cfg, mid).output[0].save()

        t_seq = time_fn(sequential, warmup=warmup, runs=runs, name=f"vllm_seq_B={B}")
        t_bat = time_fn(batched, warmup=warmup, runs=runs, name=f"vllm_bat_B={B}")
        speedup = t_seq.mean / t_bat.mean if t_bat.mean > 0 else float("inf")
        batch_results[str(B)] = {
            "sequential": t_seq.as_dict(), "batched": t_bat.as_dict(),
            "speedup": round(speedup, 3),
        }
        print(f"    B={B}: seq={t_seq.mean:.2f}ms  bat={t_bat.mean:.2f}ms  "
              f"speedup={speedup:.2f}x")

    results["sequential_vs_batched"] = batch_results

    # --- 6d: HOL blocking ---
    print("  [6d] HOL blocking")
    B = 8
    if B > len(prompts):
        B = len(prompts)

    W_enc = torch.randn(hidden_dim, 4 * hidden_dim, device=device, dtype=dtype)
    W_dec = torch.randn(4 * hidden_dim, hidden_dim, device=device, dtype=dtype)

    hol_results = {}
    for weight in ["save_only", "heavy"]:
        if weight == "save_only":
            def body(m, c):
                get_layer(m, c, mid).output[0].save()
        else:
            def body(m, c):
                h = get_layer(m, c, mid).output[0]
                enc = torch.relu(h @ W_enc)
                dec = enc @ W_dec
                get_layer(m, c, mid).output[0][:] = dec
                m.logits.output.save()

        def seq_fn(b=B, _body=body):
            for p in prompts[:b]:
                with vllm_model.trace(p, temperature=0.0, top_p=1):
                    _body(vllm_model, cfg)

        def bat_fn(b=B, _body=body):
            with vllm_model.trace(temperature=0.0, top_p=1) as tracer:
                for p in prompts[:b]:
                    with tracer.invoke(p):
                        _body(vllm_model, cfg)

        t_seq = time_fn(seq_fn, warmup=warmup, runs=runs, name=f"vllm_{weight}_seq")
        t_bat = time_fn(bat_fn, warmup=warmup, runs=runs, name=f"vllm_{weight}_bat")
        speedup = t_seq.mean / t_bat.mean if t_bat.mean > 0 else float("inf")
        hol_results[weight] = {
            "sequential": t_seq.as_dict(), "batched": t_bat.as_dict(),
            "speedup": round(speedup, 3),
        }
        print(f"    {weight}: speedup={speedup:.2f}x")

    results["hol_blocking"] = hol_results

    return results


# ---------------------------------------------------------------------------
# Output: JSON + console summary
# ---------------------------------------------------------------------------


def write_json(all_results: dict, output_dir: Path, backend: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"benchmark_batching_{backend}.json"
    with open(fname, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {fname}")


def print_summary(all_results: dict):
    print("\n" + "=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    meta = all_results.get("metadata", {})
    print(f"Model: {meta.get('model')}  Backend: {meta.get('backend')}  "
          f"Device: {meta.get('device')}")
    print()

    scenarios = all_results.get("scenarios", {})

    if "hook_overhead" in scenarios:
        s = scenarios["hook_overhead"]
        print("[1] Hook Overhead")
        print(f"    Raw: {s['raw']['mean_ms']:.2f}ms  "
              f"Empty: {s['empty_trace']['mean_ms']:.2f}ms ({s['overhead_empty']:.2f}x)  "
              f"Single: {s['single_save']['mean_ms']:.2f}ms ({s['overhead_single']:.2f}x)")

    if "mediator_scaling" in scenarios:
        s = scenarios["mediator_scaling"]
        print("[2] Mediator Scaling")
        for k in sorted((k for k in s if k != "regression"), key=int):
            print(f"    M={k:>2}: {s[k]['mean_ms']:.2f}ms")
        if "regression" in s:
            r = s["regression"]
            print(f"    Slope: {r['slope_ms_per_mediator']:.2f} ms/mediator  R²={r['r_squared']:.4f}")

    if "sequential_vs_batched" in scenarios:
        s = scenarios["sequential_vs_batched"]
        print("[3] Batching Overhead (raw vs nnsight)")
        for k in sorted(s, key=int):
            d = s[k]
            print(f"    B={k:>2}: raw={d['raw_batched']['mean_ms']:.1f}ms  "
                  f"nn={d['nnsight_batched']['mean_ms']:.1f}ms  "
                  f"overhead={d['overhead_ratio']:.2f}x (+{d['overhead_abs_ms']:.1f}ms)  "
                  f"thr: {d['throughput_raw']:.0f} vs {d['throughput_nn']:.0f} req/s")

    if "hol_blocking" in scenarios:
        s = scenarios["hol_blocking"]
        print("[4] HOL Blocking (B=8)")
        for w in ["save_only", "light", "medium", "heavy"]:
            if w in s:
                print(f"    {w:<12}: speedup={s[w]['speedup']:.2f}x")

    if "cuda_sync" in scenarios:
        s = scenarios["cuda_sync"]
        print("[5] CUDA Sync Impact")
        print(f"    save_only: {s['save_only']['mean_ms']:.2f}ms  "
              f"value_read: {s['value_read']['mean_ms']:.2f}ms ({s['delta_read_pct']:+.1f}%)  "
              f"in_place: {s['in_place_modify']['mean_ms']:.2f}ms ({s['delta_inplace_pct']:+.1f}%)")

    if "vllm" in scenarios and scenarios["vllm"] is not None:
        s = scenarios["vllm"]
        print("[6] vLLM Backend")
        if "hook_overhead" in s:
            ho = s["hook_overhead"]
            print(f"    Overhead: {ho['overhead_ratio']:.2f}x")
        if "sequential_vs_batched" in s:
            for k in sorted(s["sequential_vs_batched"], key=int):
                print(f"    B={k:>2}: speedup={s['sequential_vs_batched'][k]['speedup']:.2f}x")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Plotting (optional, requires matplotlib)
# ---------------------------------------------------------------------------


def generate_plots(all_results: dict, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = all_results.get("scenarios", {})

    # --- Plot 1: Mediator scaling ---
    if "mediator_scaling" in scenarios:
        s = scenarios["mediator_scaling"]
        ms = sorted(int(k) for k in s if k != "regression")
        times = [s[str(m)]["mean_ms"] for m in ms]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ms, times, "o-", linewidth=2, markersize=8, label="Measured")

        if "regression" in s:
            r = s["regression"]
            fit = [r["slope_ms_per_mediator"] * m + r["intercept_ms"] for m in ms]
            ax.plot(ms, fit, "--", color="red", linewidth=1.5,
                    label=f"Linear fit (R²={r['r_squared']:.3f})")

        ax.set_xlabel("Number of Mediators (M)")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Scenario 2: Mediator Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "mediator_scaling.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'mediator_scaling.png'}")

    # --- Plot 2: Batching overhead (raw vs nnsight) ---
    if "sequential_vs_batched" in scenarios:
        s = scenarios["sequential_vs_batched"]
        bs = sorted(int(k) for k in s)
        thr_raw = [s[str(b)]["throughput_raw"] for b in bs]
        thr_nn = [s[str(b)]["throughput_nn"] for b in bs]
        overheads = [s[str(b)]["overhead_ratio"] for b in bs]
        abs_gaps = [s[str(b)]["overhead_abs_ms"] for b in bs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(bs, thr_raw, "o-", label="Raw batched (no hooks)", linewidth=2)
        ax1.plot(bs, thr_nn, "s-", label="NNsight batched", linewidth=2)
        ax1.set_xlabel("Batch Size (B)")
        ax1.set_ylabel("Throughput (req/s)")
        ax1.set_title("Scenario 3: Throughput Scaling")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(bs, overheads, "o-", color="tab:red", linewidth=2, label="Overhead ratio")
        ax2.set_xlabel("Batch Size (B)")
        ax2.set_ylabel("Overhead (nnsight / raw)")
        ax2.set_title("Scenario 3: Overhead Growth with Batch Size")
        ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="No overhead")
        ax2_abs = ax2.twinx()
        ax2_abs.bar([str(b) for b in bs], abs_gaps, alpha=0.3, color="tab:blue",
                    label="Abs gap (ms)")
        ax2_abs.set_ylabel("Absolute gap (ms)")
        ax2.legend(loc="upper left")
        ax2_abs.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "batching_overhead.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'batching_overhead.png'}")

    # --- Plot 3: HOL blocking speedup collapse ---
    if "hol_blocking" in scenarios:
        s = scenarios["hol_blocking"]
        weights = [w for w in ["save_only", "light", "medium", "heavy"] if w in s]
        speedups = [s[w]["speedup"] for w in weights]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(weights, speedups, color=["#4CAF50", "#8BC34A", "#FF9800", "#F44336"])
        ax.set_xlabel("Intervention Weight")
        ax.set_ylabel("Batch Speedup (seq / batched)")
        ax.set_title("Scenario 4: HOL Blocking – Speedup Collapse")
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="No speedup")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}x", ha="center", va="bottom", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / "hol_blocking.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'hol_blocking.png'}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="NNsight batching performance benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", choices=["transformers", "vllm", "both"],
                   default="transformers", help="Backend to benchmark (default: transformers)")
    p.add_argument("--model", type=str, default=None,
                   help=f"Model name (default: {DEFAULT_MODEL}, smoke: {SMOKE_MODEL})")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    p.add_argument("--runs", type=int, default=10, help="Timed iterations (default: 10)")
    p.add_argument("--scenarios", type=str, default="all",
                   help="Comma-separated scenario numbers or 'all' (default: all)")
    p.add_argument("--smoke", action="store_true",
                   help=f"Quick run: {SMOKE_MODEL}, warmup=1, runs=3")
    p.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    p.add_argument("--output-dir", type=str, default="tests/performance/results/",
                   help="Output directory for results")
    return p.parse_args()


def main():
    args = parse_args()

    # GPU check
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    # Resolve settings
    if args.smoke:
        model_name = args.model or SMOKE_MODEL
        warmup = 1
        runs = 3
    else:
        model_name = args.model or DEFAULT_MODEL
        warmup = args.warmup
        runs = args.runs

    cfg = get_model_config(model_name)

    if args.scenarios == "all":
        scenarios = ALL_SCENARIOS[:]
    else:
        scenarios = [int(s.strip()) for s in args.scenarios.split(",")]

    # Adjust for backend
    run_transformers = args.backend in ("transformers", "both")
    run_vllm = args.backend in ("vllm", "both")
    if not run_transformers:
        scenarios = [s for s in scenarios if s == 6]
    if not run_vllm:
        scenarios = [s for s in scenarios if s != 6]

    output_dir = Path(args.output_dir)

    # Header
    print("=" * 72)
    print("NNsight Batching Performance Benchmark")
    print("=" * 72)
    print(f"Model:    {model_name}")
    print(f"Backend:  {args.backend}")
    print(f"Device:   cuda ({torch.cuda.get_device_name(0)})")
    print(f"Warmup:   {warmup}    Runs: {runs}")
    print(f"Scenarios: {scenarios}")
    print()

    # Load transformers model
    model = None
    if run_transformers and any(s in scenarios for s in [1, 2, 3, 4, 5]):
        print(f"Loading {model_name} (transformers)...")
        model = load_transformers(model_name, cfg)
        print("Model loaded.\n")

    all_results = {
        "metadata": {
            "model": model_name,
            "backend": args.backend,
            "device": f"cuda ({torch.cuda.get_device_name(0)})",
            "warmup": warmup,
            "runs": runs,
            "scenarios": scenarios,
            "timestamp": datetime.now().isoformat(),
        },
        "scenarios": {},
    }

    # Run scenarios
    if 1 in scenarios and model is not None:
        all_results["scenarios"]["hook_overhead"] = scenario_1_hook_overhead(
            model, cfg, PROMPTS, warmup, runs)

    if 2 in scenarios and model is not None:
        all_results["scenarios"]["mediator_scaling"] = scenario_2_mediator_scaling(
            model, cfg, PROMPTS, warmup, runs)

    if 3 in scenarios and model is not None:
        all_results["scenarios"]["sequential_vs_batched"] = scenario_3_sequential_vs_batched(
            model, cfg, PROMPTS, warmup, runs)

    if 4 in scenarios and model is not None:
        all_results["scenarios"]["hol_blocking"] = scenario_4_hol_blocking(
            model, cfg, PROMPTS, warmup, runs)

    if 5 in scenarios and model is not None:
        all_results["scenarios"]["cuda_sync"] = scenario_5_cuda_sync(
            model, cfg, PROMPTS, warmup, runs)

    if 6 in scenarios and run_vllm:
        all_results["scenarios"]["vllm"] = scenario_6_vllm(
            model_name, cfg, PROMPTS, warmup, runs)

    # Output
    print_summary(all_results)
    write_json(all_results, output_dir, args.backend)

    if args.plot:
        print("\nGenerating plots...")
        generate_plots(all_results, output_dir)


if __name__ == "__main__":
    main()
