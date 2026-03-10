"""
Profile the .source feature for accessing intermediate forward method operations.

This profiles:
1. Source injection overhead (one-time):
   - inspect.getsource() call
   - AST parsing
   - AST transformation
   - Code compilation and exec
   - EnvoySource/OperationEnvoy creation

2. Runtime overhead (per forward pass):
   - Overhead of wrapped operations when NOT accessing any values
   - Overhead when accessing .input/.output on operations
   - Comparison: module boundary hooks vs source operation hooks

3. Scaling:
   - How does overhead scale with complexity of forward method?
"""

import ast
import inspect
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

import nnsight
from nnsight import LanguageModel

from profiler_utils import time_function, TimingResult, sync_cuda, force_gc


def cuda_sync():
    """Helper to synchronize CUDA."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_source_injection_components(device: str = "cuda") -> Dict[str, Any]:
    """Profile individual components of source injection."""
    print("\n" + "=" * 60)
    print("PROFILING SOURCE INJECTION COMPONENTS")
    print("=" * 60)
    
    results = {}
    
    # Create a model with a complex forward method
    model = LanguageModel("gpt2", device_map=device, dispatch=True)
    
    # Get the attention module's forward method for testing
    attn_module = model.transformer.h[0].attn._module
    forward_fn = attn_module.forward
    
    # 1. Profile inspect.getsource()
    print("\n[1/5] Profiling inspect.getsource()...")
    
    def profile_getsource():
        return inspect.getsource(forward_fn)
    
    result_getsource = time_function(
        profile_getsource,
        iterations=100,
        warmup=5,
        name="inspect.getsource",
    )
    results["getsource"] = result_getsource
    print(f"  {result_getsource}")
    
    # Get the source for subsequent tests
    source = textwrap.dedent(inspect.getsource(forward_fn))
    
    # 2. Profile ast.parse()
    print("\n[2/5] Profiling ast.parse()...")
    
    def profile_ast_parse():
        return ast.parse(source)
    
    result_ast_parse = time_function(
        profile_ast_parse,
        iterations=100,
        warmup=5,
        name="ast.parse",
    )
    results["ast_parse"] = result_ast_parse
    print(f"  {result_ast_parse}")
    
    # 3. Profile AST transformation
    print("\n[3/5] Profiling AST transformation (FunctionCallWrapper)...")
    
    from nnsight.intervention.inject import FunctionCallWrapper
    
    def profile_ast_transform():
        tree = ast.parse(source)
        transformer = FunctionCallWrapper("test.module")
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        return tree, transformer.line_numbers
    
    result_ast_transform = time_function(
        profile_ast_transform,
        iterations=100,
        warmup=5,
        name="ast_transform",
    )
    results["ast_transform"] = result_ast_transform
    print(f"  {result_ast_transform}")
    
    # Count how many operations are wrapped
    tree, line_numbers = profile_ast_transform()
    num_operations = len(line_numbers)
    print(f"  Operations wrapped: {num_operations}")
    results["num_operations"] = num_operations
    
    # 4. Profile code compilation
    print("\n[4/5] Profiling compile()...")
    
    def profile_compile():
        _tree = ast.parse(source)
        _transformer = FunctionCallWrapper("test.module")
        _tree = _transformer.visit(_tree)
        ast.fix_missing_locations(_tree)
        return compile(_tree, "<nnsight>", "exec")
    
    result_compile = time_function(
        profile_compile,
        iterations=100,
        warmup=5,
        name="compile",
    )
    results["compile"] = result_compile
    print(f"  {result_compile}")
    
    # 5. Profile full inject/convert
    print("\n[5/5] Profiling full inject (convert) function...")
    
    from nnsight.intervention.inject import convert as inject
    
    def dummy_wrap(fn, **kwargs):
        return fn
    
    def profile_full_inject():
        return inject(forward_fn, dummy_wrap, "test.module")
    
    result_inject = time_function(
        profile_full_inject,
        iterations=50,
        warmup=2,
        name="full_inject",
    )
    results["full_inject"] = result_inject
    print(f"  {result_inject}")
    
    # Calculate component breakdown
    total_components = (
        result_getsource.mean_time_ms +
        result_ast_parse.mean_time_ms +
        result_ast_transform.mean_time_ms +
        result_compile.mean_time_ms
    )
    
    print("\n" + "-" * 40)
    print("COMPONENT BREAKDOWN:")
    print(f"  getsource:      {result_getsource.mean_time_ms:.3f}ms ({100*result_getsource.mean_time_ms/total_components:.1f}%)")
    print(f"  ast.parse:      {result_ast_parse.mean_time_ms:.3f}ms ({100*result_ast_parse.mean_time_ms/total_components:.1f}%)")
    print(f"  ast_transform:  {result_ast_transform.mean_time_ms:.3f}ms ({100*result_ast_transform.mean_time_ms/total_components:.1f}%)")
    print(f"  compile:        {result_compile.mean_time_ms:.3f}ms ({100*result_compile.mean_time_ms/total_components:.1f}%)")
    print(f"  ---")
    print(f"  Sum components: {total_components:.3f}ms")
    print(f"  Full inject:    {result_inject.mean_time_ms:.3f}ms")
    print(f"  Exec overhead:  {result_inject.mean_time_ms - total_components:.3f}ms (exec + namespace setup)")
    
    return results


def profile_source_access_overhead(device: str = "cuda") -> Dict[str, Any]:
    """Profile the overhead of accessing .source on an Envoy."""
    print("\n" + "=" * 60)
    print("PROFILING .source ACCESS OVERHEAD")
    print("=" * 60)
    
    results = {}
    
    # We need fresh models for each test to avoid caching
    print("\n[1/3] Baseline: Accessing module without .source...")
    
    model1 = LanguageModel("gpt2", device_map=device, dispatch=True)
    input_ids = model1.tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model1._module(input_ids)
    cuda_sync()
    
    def baseline_forward():
        with torch.no_grad():
            return model1._module(input_ids)
    
    result_baseline = time_function(
        baseline_forward,
        iterations=50,
        warmup=5,
        name="baseline_forward",
    )
    results["baseline_forward"] = result_baseline
    print(f"  {result_baseline}")
    
    # 2. Profile first .source access (injection cost)
    print("\n[2/3] First .source access (includes injection)...")
    
    model2 = LanguageModel("gpt2", device_map=device, dispatch=True)
    
    def first_source_access():
        # Force fresh model each time for accurate measurement
        _model = LanguageModel("gpt2", device_map=device, dispatch=True)
        _ = _model.transformer.h[0].attn.source
        return _model
    
    # Only run a few iterations since each creates a new model
    times = []
    for i in range(5):
        force_gc()
        start = time.perf_counter()
        _ = first_source_access()
        end = time.perf_counter()
        times.append((end - start) * 1000)
        print(f"    Iteration {i+1}: {times[-1]:.2f}ms")
    
    result_first_source = TimingResult(
        name="first_source_access",
        iterations=len(times),
        mean_time_ms=sum(times) / len(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        total_time_ms=sum(times),
    )
    results["first_source_access"] = result_first_source
    print(f"  {result_first_source}")
    
    # 3. Profile subsequent .source access (should be cached)
    print("\n[3/3] Subsequent .source access (cached)...")
    
    model3 = LanguageModel("gpt2", device_map=device, dispatch=True)
    # First access to trigger injection
    _ = model3.transformer.h[0].attn.source
    
    def cached_source_access():
        return model3.transformer.h[0].attn.source
    
    result_cached_source = time_function(
        cached_source_access,
        iterations=1000,
        warmup=10,
        name="cached_source_access",
    )
    results["cached_source_access"] = result_cached_source
    print(f"  {result_cached_source}")
    
    print("\n" + "-" * 40)
    print("SUMMARY:")
    print(f"  First .source access:     {result_first_source.mean_time_ms:.2f}ms (includes model load + injection)")
    print(f"  Cached .source access:    {result_cached_source.mean_time_ms:.4f}ms")
    
    return results


def profile_source_runtime_overhead(device: str = "cuda") -> Dict[str, Any]:
    """Profile runtime overhead when forward method is wrapped with .source."""
    print("\n" + "=" * 60)
    print("PROFILING RUNTIME OVERHEAD WITH .source")
    print("=" * 60)
    
    results = {}
    
    # Create model
    model = LanguageModel("gpt2", device_map=device, dispatch=True)
    input_ids = model.tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    
    # 1. Baseline: forward without .source accessed
    print("\n[1/4] Baseline forward (no .source accessed)...")
    
    with torch.no_grad():
        _ = model._module(input_ids)
    cuda_sync()
    
    def baseline_forward():
        cuda_sync()
        with torch.no_grad():
            result = model._module(input_ids)
        cuda_sync()
        return result
    
    result_baseline = time_function(
        baseline_forward,
        iterations=50,
        warmup=5,
        name="baseline_no_source",
    )
    results["baseline_no_source"] = result_baseline
    print(f"  {result_baseline}")
    
    # 2. Access .source on attention modules (triggers injection)
    print("\n[2/4] Accessing .source on all attention modules...")
    
    for i in range(12):
        _ = model.transformer.h[i].attn.source
    
    print(f"  Injected .source on 12 attention modules")
    
    # 3. Forward with .source injected but not used in trace
    print("\n[3/4] Forward after .source injection (no trace)...")
    
    with torch.no_grad():
        _ = model._module(input_ids)
    cuda_sync()
    
    def forward_with_source_injected():
        cuda_sync()
        with torch.no_grad():
            result = model._module(input_ids)
        cuda_sync()
        return result
    
    result_source_injected = time_function(
        forward_with_source_injected,
        iterations=50,
        warmup=5,
        name="forward_with_source_injected",
    )
    results["forward_with_source_injected"] = result_source_injected
    print(f"  {result_source_injected}")
    
    # 4. Forward with .source in a trace
    print("\n[4/4] Forward with .source in trace...")
    
    # Create fresh model for trace test
    model2 = LanguageModel("gpt2", device_map=device, dispatch=True)
    
    # Access .source
    for i in range(12):
        _ = model2.transformer.h[i].attn.source
    
    def forward_with_source_in_trace():
        cuda_sync()
        with model2.trace(input_ids):
            # Just save module output, not using .source operations
            _ = model2.transformer.h[0].output.save()
        cuda_sync()
    
    result_source_in_trace = time_function(
        forward_with_source_in_trace,
        iterations=20,
        warmup=2,
        name="forward_with_source_in_trace",
    )
    results["forward_with_source_in_trace"] = result_source_in_trace
    print(f"  {result_source_in_trace}")
    
    print("\n" + "-" * 40)
    print("RUNTIME OVERHEAD SUMMARY:")
    print(f"  Baseline (no .source):         {result_baseline.mean_time_ms:.3f}ms")
    print(f"  After .source injection:       {result_source_injected.mean_time_ms:.3f}ms")
    overhead_ms = result_source_injected.mean_time_ms - result_baseline.mean_time_ms
    overhead_pct = 100 * overhead_ms / result_baseline.mean_time_ms
    print(f"  Overhead from injection:       {overhead_ms:.3f}ms ({overhead_pct:.1f}%)")
    
    return results


def profile_source_operation_access(device: str = "cuda") -> Dict[str, Any]:
    """Profile accessing .input/.output on source operations vs module boundaries."""
    print("\n" + "=" * 60)
    print("PROFILING .source OPERATION ACCESS vs MODULE BOUNDARIES")
    print("=" * 60)
    
    results = {}
    
    model = LanguageModel("gpt2", device_map=device, dispatch=True)
    input_ids = model.tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    
    # 1. Module boundary access only
    print("\n[1/3] Module boundary access (layer output)...")
    
    def module_boundary_access():
        cuda_sync()
        with model.trace(input_ids):
            _ = model.transformer.h[0].output.save()
        cuda_sync()
    
    result_module_boundary = time_function(
        module_boundary_access,
        iterations=20,
        warmup=2,
        name="module_boundary_access",
    )
    results["module_boundary_access"] = result_module_boundary
    print(f"  {result_module_boundary}")
    
    # 2. Source operation access
    print("\n[2/3] Source operation access (attention_interface output)...")
    
    # Access .source to trigger injection
    _ = model.transformer.h[0].attn.source
    
    # Print available operations
    print(f"  Available operations: {list(model.transformer.h[0].attn.source.line_numbers.keys())[:5]}...")
    
    def source_operation_access():
        cuda_sync()
        with model.trace(input_ids):
            _ = model.transformer.h[0].attn.source.attention_interface_0.output.save()
        cuda_sync()
    
    result_source_operation = time_function(
        source_operation_access,
        iterations=20,
        warmup=2,
        name="source_operation_access",
    )
    results["source_operation_access"] = result_source_operation
    print(f"  {result_source_operation}")
    
    # 3. Multiple source operations
    print("\n[3/3] Multiple source operations access...")
    
    # Inject on more layers
    for i in range(1, 6):
        _ = model.transformer.h[i].attn.source
    
    def multiple_source_operations():
        cuda_sync()
        with model.trace(input_ids):
            for i in range(6):
                _ = model.transformer.h[i].attn.source.attention_interface_0.output.save()
        cuda_sync()
    
    result_multiple_source = time_function(
        multiple_source_operations,
        iterations=20,
        warmup=2,
        name="multiple_source_operations",
    )
    results["multiple_source_operations"] = result_multiple_source
    print(f"  {result_multiple_source}")
    
    print("\n" + "-" * 40)
    print("ACCESS OVERHEAD COMPARISON:")
    print(f"  Module boundary (1 layer):     {result_module_boundary.mean_time_ms:.3f}ms")
    print(f"  Source operation (1 layer):    {result_source_operation.mean_time_ms:.3f}ms")
    print(f"  Source operations (6 layers):  {result_multiple_source.mean_time_ms:.3f}ms")
    
    overhead = result_source_operation.mean_time_ms - result_module_boundary.mean_time_ms
    print(f"\n  Additional overhead per .source op: {overhead:.3f}ms")
    
    return results


def run_source_profiles(device: str = "cuda") -> Dict[str, Any]:
    """Run all .source profiling tests."""
    print("\n" + "#" * 70)
    print("#  NNSIGHT .source FEATURE PROFILING")
    print("#" * 70)
    
    all_results = {}
    
    # 1. Profile injection components
    print("\n>>> Starting injection component profiling...")
    try:
        all_results["injection_components"] = profile_source_injection_components(device)
    except Exception as e:
        print(f"ERROR in injection component profiling: {e}")
        all_results["injection_components"] = {"error": str(e)}
    
    force_gc()
    
    # 2. Profile .source access overhead
    print("\n>>> Starting .source access overhead profiling...")
    try:
        all_results["source_access"] = profile_source_access_overhead(device)
    except Exception as e:
        print(f"ERROR in source access profiling: {e}")
        all_results["source_access"] = {"error": str(e)}
    
    force_gc()
    
    # 3. Profile runtime overhead
    print("\n>>> Starting runtime overhead profiling...")
    try:
        all_results["runtime_overhead"] = profile_source_runtime_overhead(device)
    except Exception as e:
        print(f"ERROR in runtime overhead profiling: {e}")
        all_results["runtime_overhead"] = {"error": str(e)}
    
    force_gc()
    
    # 4. Profile operation access
    print("\n>>> Starting operation access profiling...")
    try:
        all_results["operation_access"] = profile_source_operation_access(device)
    except Exception as e:
        print(f"ERROR in operation access profiling: {e}")
        all_results["operation_access"] = {"error": str(e)}
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL .source PROFILING SUMMARY")
    print("=" * 70)
    
    if "injection_components" in all_results and "error" not in all_results["injection_components"]:
        ic = all_results["injection_components"]
        print("\nInjection Costs (one-time per module):")
        print(f"  Full inject:       {ic['full_inject'].mean_time_ms:.3f}ms")
        print(f"  Operations found:  {ic['num_operations']}")
        print(f"  Breakdown:")
        print(f"    getsource:       {ic['getsource'].mean_time_ms:.3f}ms")
        print(f"    ast.parse:       {ic['ast_parse'].mean_time_ms:.3f}ms")
        print(f"    ast_transform:   {ic['ast_transform'].mean_time_ms:.3f}ms")
        print(f"    compile:         {ic['compile'].mean_time_ms:.3f}ms")
    
    if "runtime_overhead" in all_results and "error" not in all_results["runtime_overhead"]:
        ro = all_results["runtime_overhead"]
        baseline = ro["baseline_no_source"].mean_time_ms
        injected = ro["forward_with_source_injected"].mean_time_ms
        overhead = injected - baseline
        print(f"\nRuntime Overhead (after injection):")
        print(f"  Baseline forward:  {baseline:.3f}ms")
        print(f"  With .source:      {injected:.3f}ms")
        print(f"  Overhead:          {overhead:.3f}ms ({100*overhead/baseline:.1f}%)")
    
    if "operation_access" in all_results and "error" not in all_results["operation_access"]:
        oa = all_results["operation_access"]
        print(f"\nOperation Access:")
        print(f"  Module boundary:   {oa['module_boundary_access'].mean_time_ms:.3f}ms")
        print(f"  Source operation:  {oa['source_operation_access'].mean_time_ms:.3f}ms")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile NNsight .source feature")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    args = parser.parse_args()
    
    results = run_source_profiles(args.device)
