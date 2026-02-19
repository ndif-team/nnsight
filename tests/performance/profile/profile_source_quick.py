"""
Quick profiling of NNsight .source feature.

Focused on key metrics without excessive iteration.
"""

import ast
import inspect
import textwrap
import time
from typing import Any, Dict

import torch
import torch.nn as nn

import nnsight
from nnsight import LanguageModel


def cuda_sync():
    """Helper to synchronize CUDA."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def force_gc():
    """Force garbage collection."""
    import gc
    for _ in range(3):
        gc.collect()


def time_fn(fn, iterations=10, warmup=2, name=""):
    """Simple timing function."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    cuda_sync()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        cuda_sync()
        times.append((time.perf_counter() - start) * 1000)
    
    mean = sum(times) / len(times)
    return {
        "name": name,
        "mean_ms": mean,
        "min_ms": min(times),
        "max_ms": max(times),
        "iterations": iterations,
    }


def profile_injection_components(device: str = "cuda"):
    """Profile individual injection components."""
    print("\n" + "=" * 60)
    print("INJECTION COMPONENT PROFILING")
    print("=" * 60)
    
    # Create model
    print("\nLoading model...")
    model = LanguageModel("gpt2", device_map=device, dispatch=True)
    
    # Get attention module's forward for testing
    attn_module = model.transformer.h[0].attn._module
    forward_fn = attn_module.forward
    
    # 1. getsource
    print("\n[1/4] inspect.getsource()...")
    result_getsource = time_fn(
        lambda: inspect.getsource(forward_fn),
        iterations=50,
        warmup=5,
        name="getsource"
    )
    print(f"  {result_getsource['mean_ms']:.3f}ms")
    
    source = textwrap.dedent(inspect.getsource(forward_fn))
    
    # 2. ast.parse
    print("\n[2/4] ast.parse()...")
    result_ast = time_fn(
        lambda: ast.parse(source),
        iterations=50,
        warmup=5,
        name="ast_parse"
    )
    print(f"  {result_ast['mean_ms']:.3f}ms")
    
    # 3. AST transformation
    print("\n[3/4] AST transformation...")
    from nnsight.intervention.inject import FunctionCallWrapper
    
    def do_transform():
        tree = ast.parse(source)
        transformer = FunctionCallWrapper("test")
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        return tree
    
    result_transform = time_fn(do_transform, iterations=50, warmup=5, name="transform")
    print(f"  {result_transform['mean_ms']:.3f}ms")
    
    # Count operations
    tree = do_transform()
    from nnsight.intervention.inject import FunctionCallWrapper
    temp_transformer = FunctionCallWrapper("test")
    ast.parse(source)
    temp_transformer.visit(ast.parse(source))
    num_ops = len(temp_transformer.line_numbers)
    print(f"  Operations found: {num_ops}")
    
    # 4. Full inject
    print("\n[4/4] Full inject (convert)...")
    from nnsight.intervention.inject import convert as inject
    
    def dummy_wrap(fn, **kwargs):
        return fn
    
    result_inject = time_fn(
        lambda: inject(forward_fn, dummy_wrap, "test"),
        iterations=10,
        warmup=1,
        name="full_inject"
    )
    print(f"  {result_inject['mean_ms']:.3f}ms")
    
    print("\n" + "-" * 40)
    total_parts = result_getsource['mean_ms'] + result_ast['mean_ms'] + result_transform['mean_ms']
    print(f"Component breakdown:")
    print(f"  getsource:    {result_getsource['mean_ms']:.3f}ms ({100*result_getsource['mean_ms']/total_parts:.0f}%)")
    print(f"  ast.parse:    {result_ast['mean_ms']:.3f}ms ({100*result_ast['mean_ms']/total_parts:.0f}%)")
    print(f"  transform:    {result_transform['mean_ms']:.3f}ms ({100*result_transform['mean_ms']/total_parts:.0f}%)")
    print(f"  Sum:          {total_parts:.3f}ms")
    print(f"  Full inject:  {result_inject['mean_ms']:.3f}ms")
    print(f"  Overhead:     {result_inject['mean_ms'] - total_parts:.3f}ms (compile + exec)")
    
    return {
        "getsource": result_getsource,
        "ast_parse": result_ast,
        "transform": result_transform,
        "full_inject": result_inject,
        "num_operations": num_ops,
    }


def profile_runtime_overhead(device: str = "cuda"):
    """Profile runtime overhead from .source injection."""
    print("\n" + "=" * 60)
    print("RUNTIME OVERHEAD PROFILING")
    print("=" * 60)
    
    print("\nLoading model...")
    model = LanguageModel("gpt2", device_map=device, dispatch=True)
    input_ids = model.tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    
    # Baseline
    print("\n[1/4] Baseline forward (no .source)...")
    with torch.no_grad():
        _ = model._module(input_ids)
    cuda_sync()
    
    def baseline_forward():
        with torch.no_grad():
            return model._module(input_ids)
    
    result_baseline = time_fn(baseline_forward, iterations=20, warmup=3, name="baseline")
    print(f"  {result_baseline['mean_ms']:.3f}ms")
    
    # Inject .source on all attention modules
    print("\n[2/4] Injecting .source on all 12 attention modules...")
    start = time.perf_counter()
    for i in range(12):
        _ = model.transformer.h[i].attn.source
    inject_time = (time.perf_counter() - start) * 1000
    print(f"  Injection time: {inject_time:.1f}ms")
    
    # Forward after injection (no trace)
    print("\n[3/4] Forward after .source injection (no trace)...")
    cuda_sync()
    
    result_after_inject = time_fn(baseline_forward, iterations=20, warmup=3, name="after_inject")
    print(f"  {result_after_inject['mean_ms']:.3f}ms")
    
    # Forward in trace
    print("\n[4/4] Forward in trace (not using .source ops)...")
    
    def trace_forward():
        with model.trace(input_ids):
            _ = model.transformer.h[0].output.save()
    
    result_trace = time_fn(trace_forward, iterations=10, warmup=1, name="trace")
    print(f"  {result_trace['mean_ms']:.3f}ms")
    
    print("\n" + "-" * 40)
    overhead = result_after_inject['mean_ms'] - result_baseline['mean_ms']
    print(f"RUNTIME OVERHEAD SUMMARY:")
    print(f"  Baseline forward:      {result_baseline['mean_ms']:.3f}ms")
    print(f"  After .source inject:  {result_after_inject['mean_ms']:.3f}ms")
    print(f"  Overhead:              {overhead:.3f}ms ({100*overhead/result_baseline['mean_ms']:.1f}%)")
    
    return {
        "baseline": result_baseline,
        "after_inject": result_after_inject,
        "trace": result_trace,
        "injection_time_ms": inject_time,
        "overhead_ms": overhead,
    }


def profile_operation_access(device: str = "cuda"):
    """Profile accessing .source operations vs module boundaries."""
    print("\n" + "=" * 60)
    print("OPERATION ACCESS PROFILING")
    print("=" * 60)
    
    print("\nLoading model...")
    model = LanguageModel("gpt2", device_map=device, dispatch=True)
    input_ids = model.tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    
    # Module boundary only
    print("\n[1/3] Module boundary access (layer output)...")
    
    def module_access():
        with model.trace(input_ids):
            _ = model.transformer.h[0].output.save()
    
    result_module = time_fn(module_access, iterations=10, warmup=1, name="module_boundary")
    print(f"  {result_module['mean_ms']:.3f}ms")
    
    # Access .source
    print("\n[2/3] Single .source operation access...")
    _ = model.transformer.h[0].attn.source
    
    # List operations
    ops = list(model.transformer.h[0].attn.source.line_numbers.keys())[:5]
    print(f"  Available ops: {ops}")
    
    def source_access():
        with model.trace(input_ids):
            _ = model.transformer.h[0].attn.source.attention_interface_0.output.save()
    
    result_source = time_fn(source_access, iterations=10, warmup=1, name="source_op")
    print(f"  {result_source['mean_ms']:.3f}ms")
    
    # Multiple source ops
    print("\n[3/3] Multiple .source operations (6 layers)...")
    for i in range(1, 6):
        _ = model.transformer.h[i].attn.source
    
    def multi_source():
        with model.trace(input_ids):
            for i in range(6):
                _ = model.transformer.h[i].attn.source.attention_interface_0.output.save()
    
    result_multi = time_fn(multi_source, iterations=10, warmup=1, name="multi_source")
    print(f"  {result_multi['mean_ms']:.3f}ms")
    
    print("\n" + "-" * 40)
    overhead = result_source['mean_ms'] - result_module['mean_ms']
    print(f"ACCESS OVERHEAD SUMMARY:")
    print(f"  Module boundary:         {result_module['mean_ms']:.3f}ms")
    print(f"  Single .source op:       {result_source['mean_ms']:.3f}ms")
    print(f"  6 .source ops:           {result_multi['mean_ms']:.3f}ms")
    print(f"  Overhead per .source:    {overhead:.3f}ms")
    
    return {
        "module_boundary": result_module,
        "source_op": result_source,
        "multi_source": result_multi,
        "overhead_per_source_ms": overhead,
    }


def run_quick_profile(device: str = "cuda"):
    """Run all quick profiling tests."""
    print("\n" + "#" * 70)
    print("#  NNSIGHT .source QUICK PROFILING")
    print("#" * 70)
    
    results = {}
    
    # 1. Injection components
    print("\n>>> Injection component profiling...")
    try:
        results["injection"] = profile_injection_components(device)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["injection"] = {"error": str(e)}
    
    force_gc()
    
    # 2. Runtime overhead
    print("\n>>> Runtime overhead profiling...")
    try:
        results["runtime"] = profile_runtime_overhead(device)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["runtime"] = {"error": str(e)}
    
    force_gc()
    
    # 3. Operation access
    print("\n>>> Operation access profiling...")
    try:
        results["operation"] = profile_operation_access(device)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["operation"] = {"error": str(e)}
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL .source PROFILING SUMMARY")
    print("=" * 70)
    
    if "injection" in results and "error" not in results["injection"]:
        inj = results["injection"]
        print(f"\n1. INJECTION COST (one-time per module):")
        print(f"   Full inject:    {inj['full_inject']['mean_ms']:.2f}ms")
        print(f"   Operations:     {inj['num_operations']}")
    
    if "runtime" in results and "error" not in results["runtime"]:
        rt = results["runtime"]
        print(f"\n2. RUNTIME OVERHEAD (per forward pass):")
        print(f"   Baseline:       {rt['baseline']['mean_ms']:.2f}ms")
        print(f"   With .source:   {rt['after_inject']['mean_ms']:.2f}ms")
        print(f"   Overhead:       {rt['overhead_ms']:.2f}ms ({100*rt['overhead_ms']/rt['baseline']['mean_ms']:.1f}%)")
    
    if "operation" in results and "error" not in results["operation"]:
        op = results["operation"]
        print(f"\n3. OPERATION ACCESS (per trace):")
        print(f"   Module only:    {op['module_boundary']['mean_ms']:.2f}ms")
        print(f"   With .source:   {op['source_op']['mean_ms']:.2f}ms")
        print(f"   Additional:     {op['overhead_per_source_ms']:.2f}ms per .source access")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_quick_profile(args.device)
