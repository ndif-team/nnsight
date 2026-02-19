"""
Profile the Envoy system and module wrapping in NNsight.

This module profiles:
1. Envoy tree construction overhead
2. Module wrapping and hook installation
3. Attribute access patterns
4. Source tracing overhead (.source property)
5. Model dispatch overhead

These correspond to NNsight's Envoy system as documented
in NNsight.md Section 4.
"""

import gc
import time
from typing import Dict, List, Any
import weakref

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


def create_models_of_varying_depth():
    """Create models with varying depths for testing."""

    class LayerBlock(nn.Module):
        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.activation = nn.GELU()

        def forward(self, x):
            residual = x
            x = self.linear1(x)
            x = self.norm(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x + residual

    class DeepModel(nn.Module):
        def __init__(self, num_layers: int, hidden_dim: int = 64):
            super().__init__()
            self.embed = nn.Linear(hidden_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [LayerBlock(hidden_dim) for _ in range(num_layers)]
            )
            self.head = nn.Linear(hidden_dim, 10)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    return {
        4: DeepModel(4),
        8: DeepModel(8),
        12: DeepModel(12),
        24: DeepModel(24),
        48: DeepModel(48),
    }


def count_modules(model: nn.Module) -> int:
    """Count total modules in a model."""
    return sum(1 for _ in model.modules())


def run_envoy_profiles(device: str = "cuda") -> Dict[str, Any]:
    """
    Run comprehensive profiling of NNsight's Envoy system.

    Returns:
        Dictionary with profiling results
    """
    import nnsight
    from nnsight import CONFIG

    results = {}
    timer = InstrumentedTimer()

    print("\n" + "=" * 60)
    print("PROFILING: Envoy System")
    print("=" * 60)

    # -------------------------------------------------------------
    # Profile 1: Envoy tree construction overhead
    # -------------------------------------------------------------
    print("\n1. Envoy tree construction overhead...")

    models = create_models_of_varying_depth()

    construction_results = {}
    for num_layers, model in models.items():
        model = model.to(device)
        num_modules = count_modules(model)

        def construct_wrapper(m=model):
            return nnsight.NNsight(m)

        result = time_function(
            construct_wrapper,
            iterations=10,
            warmup=2,
            name=f"envoy_construction_{num_layers}_layers",
        )
        construction_results[num_layers] = {
            "timing": result,
            "num_modules": num_modules,
            "time_per_module": result.mean_time_ms / num_modules,
        }
        print(
            f"   {num_layers} layers ({num_modules} modules): {result.mean_time_ms:.3f}ms "
            f"({result.mean_time_ms/num_modules:.4f}ms/module)"
        )

    results["envoy_construction"] = construction_results

    # -------------------------------------------------------------
    # Profile 2: Module wrapping overhead (hooks only)
    # -------------------------------------------------------------
    print("\n2. Module wrapping overhead (per module)...")

    simple_module = nn.Linear(64, 64).to(device)

    def wrap_module_only():
        # Simulate what wrap_module does without full Envoy construction
        original_forward = type(simple_module).forward

        def nnsight_forward(*args, **kwargs):
            return original_forward(simple_module, *args, **kwargs)

        simple_module.forward = nnsight_forward

        def input_hook(module, args, kwargs):
            return args, kwargs

        def output_hook(module, _, output):
            return output

        h1 = simple_module.register_forward_pre_hook(input_hook, with_kwargs=True)
        h2 = simple_module.register_forward_hook(output_hook)

        # Cleanup
        h1.remove()
        h2.remove()
        simple_module.forward = original_forward

    result_wrap = time_function(
        wrap_module_only,
        iterations=100,
        warmup=10,
        name="module_wrapping",
    )
    results["module_wrapping"] = result_wrap
    print(f"   Per-module wrapping: {result_wrap}")

    # -------------------------------------------------------------
    # Profile 3: Attribute access patterns
    # -------------------------------------------------------------
    print("\n3. Attribute access patterns...")

    test_model = models[12].to(device)
    wrapper = nnsight.NNsight(test_model)
    input_tensor = torch.randn(1, 64, device=device)

    # Direct child access
    def access_child():
        _ = wrapper.layers

    result_child = time_function(
        access_child,
        iterations=10000,
        warmup=1000,
        name="child_access",
    )
    results["child_access"] = result_child
    print(f"   Direct child access: {result_child}")

    # Deep path access
    def access_deep_path():
        _ = wrapper.layers[5].linear1

    result_deep = time_function(
        access_deep_path,
        iterations=10000,
        warmup=1000,
        name="deep_path_access",
    )
    results["deep_path_access"] = result_deep
    print(f"   Deep path access (.layers[5].linear1): {result_deep}")

    # Module list indexing
    def access_list_index():
        _ = wrapper.layers[0]
        _ = wrapper.layers[5]
        _ = wrapper.layers[11]

    result_index = time_function(
        access_list_index,
        iterations=10000,
        warmup=1000,
        name="list_indexing",
    )
    results["list_indexing"] = result_index
    print(f"   ModuleList indexing (3 indices): {result_index}")

    # Access underlying module attribute
    def access_module_attr():
        _ = wrapper.layers[0].linear1.weight

    result_attr = time_function(
        access_module_attr,
        iterations=10000,
        warmup=1000,
        name="module_attribute",
    )
    results["module_attribute"] = result_attr
    print(f"   Module attribute access (.weight): {result_attr}")

    # -------------------------------------------------------------
    # Profile 4: .output/.input access during trace
    # -------------------------------------------------------------
    print("\n4. Value access during trace...")

    # First, establish a baseline of just running the model
    def baseline_forward():
        with torch.no_grad():
            _ = test_model(input_tensor)

    result_baseline = time_function(
        baseline_forward,
        iterations=20,
        warmup=5,
        name="baseline_forward",
    )
    results["baseline_forward"] = result_baseline
    print(f"   Baseline forward (no trace): {result_baseline}")

    # Trace with no interventions
    def trace_no_intervention():
        with wrapper.trace(input_tensor):
            pass

    result_no_int = time_function(
        trace_no_intervention,
        iterations=20,
        warmup=5,
        name="trace_no_intervention",
    )
    results["trace_no_intervention"] = result_no_int
    trace_overhead = result_no_int.mean_time_ms - result_baseline.mean_time_ms
    print(
        f"   Trace (no interventions): {result_no_int} (overhead: {trace_overhead:.3f}ms)"
    )

    # Trace with single .output.save()
    def trace_single_output():
        with wrapper.trace(input_tensor):
            _ = wrapper.layers[0].output.save()

    result_single = time_function(
        trace_single_output,
        iterations=20,
        warmup=5,
        name="trace_single_output",
    )
    results["trace_single_output"] = result_single
    single_overhead = result_single.mean_time_ms - result_no_int.mean_time_ms
    print(f"   Trace (1 output.save()): {result_single} (+{single_overhead:.3f}ms)")

    # Trace with multiple .output.save()
    def trace_multiple_outputs():
        with wrapper.trace(input_tensor):
            for i in range(6):
                _ = wrapper.layers[i].output.save()

    result_multiple = time_function(
        trace_multiple_outputs,
        iterations=20,
        warmup=5,
        name="trace_multiple_outputs",
    )
    results["trace_multiple_outputs"] = result_multiple
    per_intervention = (result_multiple.mean_time_ms - result_no_int.mean_time_ms) / 6
    print(
        f"   Trace (6 output.save()): {result_multiple} ({per_intervention:.3f}ms/intervention)"
    )

    results["per_intervention_overhead_ms"] = per_intervention

    # -------------------------------------------------------------
    # Profile 5: Source tracing overhead
    # -------------------------------------------------------------
    print("\n5. Source tracing (.source) overhead...")

    # Note: .source rewrites the forward method using AST injection
    # This is a one-time cost per module

    fresh_model = models[4].to(device)
    fresh_wrapper = nnsight.NNsight(fresh_model)

    def access_source():
        _ = fresh_wrapper.layers[0].source

    result_source_first = time_function(
        access_source,
        iterations=1,
        warmup=0,
        name="source_first_access",
    )
    results["source_first_access"] = result_source_first
    print(
        f"   First .source access (AST injection): {result_source_first.mean_time_ms:.3f}ms"
    )

    # Subsequent access (cached)
    def access_source_cached():
        _ = fresh_wrapper.layers[0].source

    result_source_cached = time_function(
        access_source_cached,
        iterations=100,
        warmup=10,
        name="source_cached_access",
    )
    results["source_cached_access"] = result_source_cached
    print(f"   Cached .source access: {result_source_cached}")

    # -------------------------------------------------------------
    # Profile 6: Dispatch overhead
    # -------------------------------------------------------------
    print("\n6. Model dispatch overhead...")

    # Create model with dispatch=True (meta device loading)
    # Note: This is more relevant for large HuggingFace models

    def create_and_dispatch():
        m = models[12].to("cpu")
        w = nnsight.NNsight(m)
        m.to(device)
        # Simulate what _update does
        for child_env in w._children:
            pass
        return w

    result_dispatch = time_function(
        create_and_dispatch,
        iterations=5,
        warmup=1,
        name="model_dispatch_simulation",
    )
    results["model_dispatch"] = result_dispatch
    print(f"   Model dispatch simulation: {result_dispatch}")

    # -------------------------------------------------------------
    # Profile 7: Memory overhead of Envoy tree
    # -------------------------------------------------------------
    print("\n7. Memory overhead of Envoy tree...")

    import tracemalloc

    for num_layers in [12, 48]:
        test_model = models[num_layers].to(device)
        num_modules = count_modules(test_model)

        force_gc()
        tracemalloc.start()

        wrapper = nnsight.NNsight(test_model)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_per_module = current / num_modules / 1024  # KB
        results[f"memory_{num_layers}_layers"] = {
            "total_kb": current / 1024,
            "peak_kb": peak / 1024,
            "num_modules": num_modules,
            "kb_per_module": memory_per_module,
        }
        print(
            f"   {num_layers} layers ({num_modules} modules): "
            f"{current/1024:.1f}KB ({memory_per_module:.2f}KB/module)"
        )

    # -------------------------------------------------------------
    # Summary analysis
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("ENVOY SYSTEM SUMMARY")
    print("-" * 60)

    print(f"\nEnvoy construction scaling:")
    for num_layers, data in construction_results.items():
        print(
            f"  {num_layers:2d} layers: {data['timing'].mean_time_ms:.2f}ms "
            f"({data['time_per_module']:.4f}ms/module)"
        )

    print(f"\nAttribute access costs:")
    print(f"  - Direct child:  {result_child.mean_time_ms:.6f}ms")
    print(f"  - Deep path:     {result_deep.mean_time_ms:.6f}ms")
    print(f"  - List indexing: {result_index.mean_time_ms:.6f}ms")
    print(f"  - Module attr:   {result_attr.mean_time_ms:.6f}ms")

    print(f"\nIntervention overhead:")
    print(f"  - Trace setup:        {trace_overhead:.3f}ms")
    print(f"  - Per intervention:   {per_intervention:.3f}ms")
    print(
        f"  - .source injection:  {result_source_first.mean_time_ms:.3f}ms (one-time)"
    )

    return results


def get_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations based on profiling results."""
    recommendations = []

    # Envoy construction scaling
    if "envoy_construction" in results:
        construction = results["envoy_construction"]
        times_per_module = [d["time_per_module"] for d in construction.values()]
        avg_time = sum(times_per_module) / len(times_per_module)

        if avg_time > 0.01:  # More than 0.01ms per module
            recommendations.append(
                f"⚠ Envoy construction takes ~{avg_time:.4f}ms per module. "
                f"For very large models (100+ modules), consider lazy Envoy construction."
            )
        else:
            recommendations.append(
                f"✓ Envoy construction is efficient ({avg_time:.4f}ms/module)."
            )

    # Per-intervention overhead
    if "per_intervention_overhead_ms" in results:
        overhead = results["per_intervention_overhead_ms"]
        if overhead > 0.5:
            recommendations.append(
                f"⚠ Each intervention adds ~{overhead:.2f}ms overhead. "
                f"Batch interventions where possible (e.g., one .save() for a list of outputs)."
            )
        else:
            recommendations.append(
                f"✓ Per-intervention overhead is acceptable ({overhead:.2f}ms)."
            )

    # Source tracing
    if "source_first_access" in results:
        source_time = results["source_first_access"].mean_time_ms
        if source_time > 5.0:
            recommendations.append(
                f"⚠ .source AST injection takes {source_time:.1f}ms. "
                f"This is one-time per module but can add up for many .source accesses."
            )

    # Attribute access
    if "deep_path_access" in results:
        deep_time = results["deep_path_access"].mean_time_ms
        if deep_time > 0.001:  # More than 1 microsecond
            recommendations.append(
                f"⚠ Deep path access takes {deep_time*1000:.2f}μs. "
                f"Consider caching Envoy references for hot paths."
            )

    # Memory overhead
    for key, value in results.items():
        if key.startswith("memory_") and isinstance(value, dict):
            kb_per_module = value.get("kb_per_module", 0)
            if kb_per_module > 1.0:  # More than 1KB per module
                recommendations.append(
                    f"⚠ Envoy tree uses {kb_per_module:.2f}KB per module. "
                    f"For very large models, this may be significant."
                )
            break

    return recommendations


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = run_envoy_profiles(device)

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    for rec in get_recommendations(results):
        print(f"\n{rec}")
