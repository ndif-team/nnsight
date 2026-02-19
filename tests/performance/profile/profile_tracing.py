"""
Profile the tracing phase of NNsight.

This module profiles:
1. Source code capture (walking call stack, getting source lines)
2. AST parsing and with-block detection
3. Code compilation
4. Trace caching effectiveness

These correspond to Steps 1-3 of NNsight's tracing process as documented
in NNsight.md Section 2.
"""

import gc
import time
import ast
import inspect
from typing import Dict, List, Any

import torch
import torch.nn as nn

from profiler_utils import (
    TimingResult,
    ProfileResult,
    InstrumentedTimer,
    time_function,
    profile_function,
    force_gc,
    sync_cuda,
)


def create_simple_model() -> nn.Module:
    """Create a simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(20, 5)

        def forward(self, x):
            return self.layer2(self.relu(self.layer1(x)))

    return SimpleModel()


def run_tracing_profiles(device: str = "cuda") -> Dict[str, Any]:
    """
    Run comprehensive profiling of NNsight's tracing phase.

    Returns:
        Dictionary with profiling results
    """
    import nnsight
    from nnsight import CONFIG

    results = {}
    timer = InstrumentedTimer()

    import sys

    print("\n" + "=" * 60, flush=True)
    print("PROFILING: Tracing Phase", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()

    # Create model
    model = create_simple_model().to(device)
    wrapper = nnsight.NNsight(model)

    input_tensor = torch.randn(1, 10, device=device)

    # -------------------------------------------------------------
    # Profile 1: Full trace overhead (first trace - no cache)
    # -------------------------------------------------------------
    print("\n1. Full trace overhead (uncached)...")

    # Disable caching to measure full overhead
    original_cache_setting = CONFIG.APP.TRACE_CACHING
    CONFIG.APP.TRACE_CACHING = False

    def full_trace_uncached():
        with wrapper.trace(input_tensor):
            _ = wrapper.layer2.output.save()

    result_uncached = time_function(
        full_trace_uncached,
        iterations=5,
        warmup=1,
        name="full_trace_uncached",
    )
    results["full_trace_uncached"] = result_uncached
    print(f"   {result_uncached}")

    # -------------------------------------------------------------
    # Profile 2: Full trace overhead (cached)
    # -------------------------------------------------------------
    print("\n2. Full trace overhead (cached)...")

    CONFIG.APP.TRACE_CACHING = True

    # First call to populate cache
    with wrapper.trace(input_tensor):
        _ = wrapper.layer2.output.save()

    def full_trace_cached():
        with wrapper.trace(input_tensor):
            _ = wrapper.layer2.output.save()

    result_cached = time_function(
        full_trace_cached,
        iterations=5,
        warmup=1,
        name="full_trace_cached",
    )
    results["full_trace_cached"] = result_cached
    print(f"   {result_cached}")

    # Caching improvement ratio
    cache_improvement = result_uncached.mean_time_ms / result_cached.mean_time_ms
    results["cache_improvement_ratio"] = cache_improvement
    print(f"   Cache improvement: {cache_improvement:.2f}x faster")

    # Restore original setting
    CONFIG.APP.TRACE_CACHING = original_cache_setting

    # -------------------------------------------------------------
    # Profile 3: AST parsing overhead
    # -------------------------------------------------------------
    print("\n3. AST parsing overhead...")

    sample_code = """
with model.trace("Hello"):
    hidden = model.transformer.h[0].output.save()
    modified = hidden * 2
    model.transformer.h[1].input = modified
    final = model.lm_head.output.save()
"""

    def ast_parse_only():
        tree = ast.parse(sample_code)
        return tree

    result_ast = time_function(
        ast_parse_only,
        iterations=100,
        warmup=10,
        name="ast_parsing",
    )
    results["ast_parsing"] = result_ast
    print(f"   {result_ast}")

    # -------------------------------------------------------------
    # Profile 4: inspect.getsourcelines overhead
    # -------------------------------------------------------------
    print("\n4. inspect.getsourcelines overhead...")

    def get_source_lines():
        frame = inspect.currentframe()
        try:
            lines, offset = inspect.getsourcelines(frame)
            return lines
        finally:
            del frame

    result_inspect = time_function(
        get_source_lines,
        iterations=100,
        warmup=10,
        name="inspect_getsourcelines",
    )
    results["inspect_getsourcelines"] = result_inspect
    print(f"   {result_inspect}")

    # -------------------------------------------------------------
    # Profile 5: Code compilation overhead
    # -------------------------------------------------------------
    print("\n5. Code compilation overhead...")

    def compile_code():
        code_str = """
def test_function():
    x = some_variable
    y = x + 1
    return y
"""
        compiled = compile(code_str, "<test>", "exec")
        return compiled

    result_compile = time_function(
        compile_code,
        iterations=100,
        warmup=10,
        name="code_compilation",
    )
    results["code_compilation"] = result_compile
    print(f"   {result_compile}")

    # -------------------------------------------------------------
    # Profile 6: Call stack walking overhead
    # -------------------------------------------------------------
    print("\n6. Call stack walking overhead...")

    def walk_call_stack():
        frame = inspect.currentframe()
        depth = 0
        while frame is not None:
            depth += 1
            frame = frame.f_back
        return depth

    result_stack = time_function(
        walk_call_stack,
        iterations=1000,
        warmup=100,
        name="call_stack_walk",
    )
    results["call_stack_walk"] = result_stack
    print(f"   {result_stack}")

    # -------------------------------------------------------------
    # Profile 7: Varying trace complexity
    # -------------------------------------------------------------
    print("\n7. Trace complexity scaling...")

    # Create a larger model for complexity testing
    class DeepModel(nn.Module):
        def __init__(self, num_layers: int = 12):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(64, 64),
                        nn.ReLU(),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final = nn.Linear(64, 10)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.final(x)

    complexity_results = {}
    for num_interventions in [1, 2, 4, 8]:
        deep_model = DeepModel(num_layers=12).to(device)
        deep_wrapper = nnsight.NNsight(deep_model)
        deep_input = torch.randn(1, 64, device=device)

        def trace_with_interventions(n=num_interventions):
            with deep_wrapper.trace(deep_input):
                for i in range(n):
                    _ = deep_wrapper.layers[i].output.save()

        result = time_function(
            trace_with_interventions,
            iterations=10,
            warmup=2,
            name=f"trace_{num_interventions}_interventions",
        )
        complexity_results[num_interventions] = result
        print(f"   {num_interventions} interventions: {result.mean_time_ms:.3f}ms")

    results["complexity_scaling"] = complexity_results

    # -------------------------------------------------------------
    # Profile 8: Multiple invokes overhead
    # -------------------------------------------------------------
    print("\n8. Multiple invokes overhead...")

    invoke_results = {}
    for num_invokes in [1, 2, 4]:

        def trace_with_invokes(n=num_invokes):
            with deep_wrapper.trace() as tracer:
                for i in range(n):
                    with tracer.invoke(deep_input):
                        _ = deep_wrapper.layers[0].output.save()

        result = time_function(
            trace_with_invokes,
            iterations=10,
            warmup=2,
            name=f"trace_{num_invokes}_invokes",
        )
        invoke_results[num_invokes] = result
        print(f"   {num_invokes} invokes: {result.mean_time_ms:.3f}ms")

    results["invoke_scaling"] = invoke_results

    # -------------------------------------------------------------
    # Summary analysis
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TRACING PHASE SUMMARY")
    print("-" * 60)

    # Calculate overhead breakdown
    total_trace = result_uncached.mean_time_ms
    ast_overhead = result_ast.mean_time_ms
    inspect_overhead = result_inspect.mean_time_ms
    compile_overhead = result_compile.mean_time_ms

    print(f"\nOverhead breakdown (uncached trace = {total_trace:.3f}ms):")
    print(
        f"  - AST parsing:        {ast_overhead:.3f}ms ({100*ast_overhead/total_trace:.1f}%)"
    )
    print(
        f"  - Source inspection:  {inspect_overhead:.3f}ms ({100*inspect_overhead/total_trace:.1f}%)"
    )
    print(
        f"  - Code compilation:   {compile_overhead:.3f}ms ({100*compile_overhead/total_trace:.1f}%)"
    )
    estimated_other = total_trace - ast_overhead - inspect_overhead - compile_overhead
    print(f"  - Other (threading, hooks, execution): ~{max(0, estimated_other):.3f}ms")

    print(f"\nCaching impact:")
    print(f"  - Uncached: {result_uncached.mean_time_ms:.3f}ms")
    print(f"  - Cached:   {result_cached.mean_time_ms:.3f}ms")
    print(
        f"  - Savings:  {result_uncached.mean_time_ms - result_cached.mean_time_ms:.3f}ms ({cache_improvement:.2f}x)"
    )

    return results


def get_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations based on profiling results."""
    recommendations = []

    # Check caching impact
    if results.get("cache_improvement_ratio", 1.0) > 1.5:
        recommendations.append(
            f"✓ TRACE_CACHING is effective ({results['cache_improvement_ratio']:.1f}x improvement). "
            f"Recommend keeping it enabled by default."
        )

    # Check AST parsing overhead
    if "ast_parsing" in results:
        ast_time = results["ast_parsing"].mean_time_ms
        if ast_time > 0.5:
            recommendations.append(
                f"⚠ AST parsing takes {ast_time:.2f}ms. Consider caching parsed ASTs."
            )

    # Check source inspection overhead
    if "inspect_getsourcelines" in results:
        inspect_time = results["inspect_getsourcelines"].mean_time_ms
        if inspect_time > 1.0:
            recommendations.append(
                f"⚠ Source inspection takes {inspect_time:.2f}ms. This is expected but can "
                f"be avoided with TRACE_CACHING."
            )

    # Check complexity scaling
    if "complexity_scaling" in results:
        times = [(n, r.mean_time_ms) for n, r in results["complexity_scaling"].items()]
        if len(times) >= 2:
            t1, t4 = times[0][1], times[-1][1]
            ratio = t4 / t1
            n1, n4 = times[0][0], times[-1][0]
            if ratio > (n4 / n1) * 1.5:
                recommendations.append(
                    f"⚠ Trace overhead scales superlinearly with interventions "
                    f"({n1}→{n4} interventions: {ratio:.1f}x time increase). "
                    f"Consider batching intervention setup."
                )
            else:
                recommendations.append(
                    f"✓ Trace overhead scales well with intervention count."
                )

    return recommendations


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = run_tracing_profiles(device)

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    for rec in get_recommendations(results):
        print(f"\n{rec}")
