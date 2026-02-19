#!/usr/bin/env python3
"""
Quick NNsight Performance Profiling

A simplified profiler that runs quickly and provides key insights.
"""

import gc
import time
import sys
import threading
import _thread
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import torch.nn as nn


@dataclass
class Timing:
    name: str
    mean_ms: float
    min_ms: float
    max_ms: float
    n: int
    
    def __str__(self):
        return f"{self.name}: {self.mean_ms:.3f}ms (min={self.min_ms:.3f}, max={self.max_ms:.3f}, n={self.n})"


def time_fn(fn, n=5, warmup=1, name=None):
    """Time a function quickly."""
    name = name or fn.__name__
    
    # Warmup
    for _ in range(warmup):
        fn()
        gc.collect()
    
    # Time
    times = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        gc.collect()
    
    return Timing(name, sum(times)/len(times), min(times), max(times), n)


def create_model(num_layers=12, hidden=64, device="cuda"):
    """Create a test model."""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
                for _ in range(num_layers)
            ])
            self.head = nn.Linear(hidden, 10)
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.head(x)
    
    return Model().to(device)


def run_quick_profile(device="cuda"):
    """Run quick profiling of NNsight."""
    import nnsight
    from nnsight import CONFIG
    
    results = {}
    
    print("=" * 60, flush=True)
    print("NNsight Quick Performance Profile", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {device}", flush=True)
    print()
    
    # =========================================================
    # 1. BASELINE: Model forward without NNsight
    # =========================================================
    print("[1/8] Model baseline...", flush=True)
    
    model = create_model(12, 64, device)
    x = torch.randn(1, 64, device=device)
    
    def baseline_forward():
        with torch.no_grad():
            return model(x)
    
    t = time_fn(baseline_forward, n=10, warmup=2, name="baseline_forward")
    results["baseline"] = t
    print(f"  {t}", flush=True)
    
    # =========================================================
    # 2. WRAPPING: NNsight wrapper creation
    # =========================================================
    print("[2/8] NNsight wrapper creation...", flush=True)
    
    def wrap_model():
        return nnsight.NNsight(model)
    
    t = time_fn(wrap_model, n=3, warmup=1, name="nnsight_wrap")
    results["wrapping"] = t
    print(f"  {t}", flush=True)
    
    wrapper = nnsight.NNsight(model)
    
    # =========================================================
    # 3. TRACE: Empty trace overhead
    # =========================================================
    print("[3/8] Empty trace overhead...", flush=True)
    
    def empty_trace():
        with wrapper.trace(x):
            pass
    
    t = time_fn(empty_trace, n=5, warmup=1, name="empty_trace")
    results["empty_trace"] = t
    print(f"  {t}", flush=True)
    
    # =========================================================
    # 4. INTERVENTION: Single .output.save()
    # =========================================================
    print("[4/8] Single intervention...", flush=True)
    
    def single_intervention():
        with wrapper.trace(x):
            _ = wrapper.layers[0].output.save()
    
    t = time_fn(single_intervention, n=5, warmup=1, name="single_intervention")
    results["single_intervention"] = t
    print(f"  {t}", flush=True)
    
    intervention_overhead = t.mean_ms - results["empty_trace"].mean_ms
    results["per_intervention_overhead"] = intervention_overhead
    print(f"  Per-intervention overhead: {intervention_overhead:.3f}ms", flush=True)
    
    # =========================================================
    # 5. SCALING: Multiple interventions
    # =========================================================
    print("[5/8] Multiple interventions...", flush=True)
    
    def six_interventions():
        with wrapper.trace(x):
            for i in range(6):
                _ = wrapper.layers[i].output.save()
    
    t = time_fn(six_interventions, n=5, warmup=1, name="six_interventions")
    results["six_interventions"] = t
    print(f"  {t}", flush=True)
    
    avg_intervention = (t.mean_ms - results["empty_trace"].mean_ms) / 6
    print(f"  Per-intervention (avg of 6): {avg_intervention:.3f}ms", flush=True)
    
    # =========================================================
    # 6. MULTI-INVOKE: Multiple invokes (requires LanguageModel for batching)
    # =========================================================
    print("[6/8] Multi-invoke (skipped - requires LanguageModel)...", flush=True)
    print("  Note: Multiple invokes require a model with _batch() implementation", flush=True)
    
    # =========================================================
    # 7. CONFIG: CROSS_INVOKER impact (tested with single invoke)
    # =========================================================
    print("[7/8] CROSS_INVOKER config impact...", flush=True)
    
    original_setting = CONFIG.APP.CROSS_INVOKER
    
    # Test with single invoke - overhead still visible in variable sharing
    def trace_with_config():
        with wrapper.trace(x):
            _ = wrapper.layers[0].output.save()
    
    CONFIG.APP.CROSS_INVOKER = False
    t_disabled = time_fn(trace_with_config, n=5, warmup=1, name="cross_invoker_disabled")
    
    CONFIG.APP.CROSS_INVOKER = True
    t_enabled = time_fn(trace_with_config, n=5, warmup=1, name="cross_invoker_enabled")
    
    CONFIG.APP.CROSS_INVOKER = original_setting
    
    results["cross_invoker_disabled"] = t_disabled
    results["cross_invoker_enabled"] = t_enabled
    print(f"  CROSS_INVOKER=False: {t_disabled.mean_ms:.3f}ms", flush=True)
    print(f"  CROSS_INVOKER=True:  {t_enabled.mean_ms:.3f}ms", flush=True)
    overhead = t_enabled.mean_ms - t_disabled.mean_ms
    print(f"  Overhead: {overhead:.3f}ms ({'+' if overhead > 0 else ''}{overhead:.3f}ms)", flush=True)
    
    # =========================================================
    # 8. CACHING: Trace caching impact
    # =========================================================
    print("[8/8] Trace caching impact...", flush=True)
    
    # Create fresh wrapper to measure uncached
    fresh_model = create_model(12, 64, device)
    
    CONFIG.APP.TRACE_CACHING = False
    fresh_wrapper = nnsight.NNsight(fresh_model)
    
    def uncached_trace():
        with fresh_wrapper.trace(x):
            _ = fresh_wrapper.layers[0].output.save()
    
    t_uncached = time_fn(uncached_trace, n=3, warmup=0, name="uncached_trace")
    
    CONFIG.APP.TRACE_CACHING = True
    fresh_wrapper2 = nnsight.NNsight(fresh_model)
    
    # First call to populate cache
    with fresh_wrapper2.trace(x):
        _ = fresh_wrapper2.layers[0].output.save()
    
    def cached_trace():
        with fresh_wrapper2.trace(x):
            _ = fresh_wrapper2.layers[0].output.save()
    
    t_cached = time_fn(cached_trace, n=3, warmup=0, name="cached_trace")
    
    results["uncached_trace"] = t_uncached
    results["cached_trace"] = t_cached
    print(f"  TRACE_CACHING=False: {t_uncached.mean_ms:.3f}ms", flush=True)
    print(f"  TRACE_CACHING=True:  {t_cached.mean_ms:.3f}ms", flush=True)
    cache_speedup = t_uncached.mean_ms / t_cached.mean_ms
    print(f"  Speedup: {cache_speedup:.2f}x", flush=True)
    
    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    
    print(f"\nOverhead breakdown:", flush=True)
    print(f"  Model baseline:         {results['baseline'].mean_ms:.3f}ms", flush=True)
    print(f"  + Empty trace:          +{results['empty_trace'].mean_ms - results['baseline'].mean_ms:.3f}ms", flush=True)
    print(f"  + Per intervention:     +{intervention_overhead:.3f}ms", flush=True)
    
    nnsight_total = results['single_intervention'].mean_ms
    baseline = results['baseline'].mean_ms
    overhead_ratio = nnsight_total / baseline
    print(f"\n  NNsight total (1 save): {nnsight_total:.3f}ms", flush=True)
    print(f"  Overhead ratio:         {overhead_ratio:.2f}x baseline", flush=True)
    
    print(f"\nConfiguration impacts:", flush=True)
    print(f"  CROSS_INVOKER overhead: {t_enabled.mean_ms - t_disabled.mean_ms:.3f}ms", flush=True)
    print(f"  TRACE_CACHING speedup:  {cache_speedup:.2f}x", flush=True)
    
    # =========================================================
    # RECOMMENDATIONS
    # =========================================================
    print("\n" + "=" * 60, flush=True)
    print("RECOMMENDATIONS", flush=True)
    print("=" * 60, flush=True)
    
    recommendations = []
    
    if cache_speedup > 1.1:
        recommendations.append(
            f"✓ TRACE_CACHING is effective ({cache_speedup:.1f}x speedup). Keep enabled."
        )
    
    cross_overhead = t_enabled.mean_ms - t_disabled.mean_ms
    if cross_overhead > 0.5:
        recommendations.append(
            f"⚠ CROSS_INVOKER adds {cross_overhead:.2f}ms. "
            f"Disable if not sharing variables between invokes."
        )
    else:
        recommendations.append(
            f"✓ CROSS_INVOKER overhead is minimal ({cross_overhead:.2f}ms)."
        )
    
    if intervention_overhead > 0.3:
        recommendations.append(
            f"⚠ Each intervention adds {intervention_overhead:.2f}ms. "
            f"Minimize intervention count for best performance."
        )
    
    if overhead_ratio > 2.0:
        recommendations.append(
            f"⚠ NNsight adds {overhead_ratio:.1f}x overhead vs bare PyTorch. "
            f"Expected for small models; ratio improves with larger models."
        )
    else:
        recommendations.append(
            f"✓ NNsight overhead is reasonable ({overhead_ratio:.1f}x baseline)."
        )
    
    for rec in recommendations:
        print(f"\n{rec}", flush=True)
    
    return results


def run_languagemodel_profile(device="cuda"):
    """Profile with a real HuggingFace model."""
    print("\n" + "=" * 60, flush=True)
    print("LanguageModel Profile (GPT-2)", flush=True)
    print("=" * 60, flush=True)
    
    try:
        from nnsight import LanguageModel, CONFIG
    except ImportError:
        print("LanguageModel not available", flush=True)
        return {}
    
    results = {}
    
    print("\n[1/4] Loading GPT-2...", flush=True)
    model = LanguageModel("openai-community/gpt2", device_map=device, dispatch=True)
    print("  Loaded!", flush=True)
    
    prompt = "The capital of France is"
    
    print("[2/4] Baseline generation...", flush=True)
    
    def baseline_generate():
        with torch.no_grad():
            tokens = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            return model._model.generate(tokens, max_new_tokens=5, pad_token_id=50256)
    
    t = time_fn(baseline_generate, n=3, warmup=1, name="baseline_generate")
    results["baseline_generate"] = t
    print(f"  {t}", flush=True)
    
    print("[3/4] NNsight trace with generation...", flush=True)
    
    def nnsight_generate():
        with model.generate(prompt, max_new_tokens=5, pad_token_id=50256) as gen:
            with gen.iter[:] as token_idx:
                _ = model.transformer.h[0].output[0].save()
    
    t = time_fn(nnsight_generate, n=3, warmup=1, name="nnsight_generate")
    results["nnsight_generate"] = t
    print(f"  {t}", flush=True)
    
    overhead = t.mean_ms - results["baseline_generate"].mean_ms
    ratio = t.mean_ms / results["baseline_generate"].mean_ms
    print(f"  Overhead: {overhead:.2f}ms ({ratio:.2f}x)", flush=True)
    
    print("[4/4] Multiple layer interventions...", flush=True)
    
    def nnsight_multi_layer():
        with model.generate(prompt, max_new_tokens=5, pad_token_id=50256) as gen:
            with gen.iter[:] as token_idx:
                for i in range(6):
                    _ = model.transformer.h[i].output[0].save()
    
    t = time_fn(nnsight_multi_layer, n=3, warmup=1, name="nnsight_multi_layer")
    results["nnsight_multi_layer"] = t
    print(f"  {t}", flush=True)
    
    per_layer = (t.mean_ms - results["nnsight_generate"].mean_ms) / 5  # 5 extra layers
    print(f"  Per-layer overhead: {per_layer:.3f}ms", flush=True)
    
    print("\n" + "-" * 60, flush=True)
    print("LanguageModel Summary:", flush=True)
    print(f"  Baseline generation: {results['baseline_generate'].mean_ms:.2f}ms", flush=True)
    print(f"  NNsight generation:  {results['nnsight_generate'].mean_ms:.2f}ms", flush=True)
    print(f"  Overhead ratio:      {ratio:.2f}x", flush=True)
    print(f"  Per-layer save:      {per_layer:.3f}ms", flush=True)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lm", action="store_true", help="Skip LanguageModel tests")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Quick profile with simple model
    run_quick_profile(device)
    
    # LanguageModel profile (optional)
    if not args.skip_lm:
        print("\n", flush=True)
        run_languagemodel_profile(device)
