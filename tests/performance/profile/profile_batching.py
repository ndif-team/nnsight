"""
Profile the batching operations in NNsight.

This module profiles:
1. Batcher.narrow() - tensor slicing for batch groups
2. Batcher.swap() - tensor replacement in batches
3. apply/applyn utility functions
4. Multi-invoke batching overhead

These correspond to NNsight's batching system as documented
in NNsight.md Section 3.2 and the batching.py implementation.
"""

import gc
import time
from typing import Dict, List, Any, Tuple

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


def run_batching_profiles(device: str = "cuda") -> Dict[str, Any]:
    """
    Run comprehensive profiling of NNsight's batching operations.

    Returns:
        Dictionary with profiling results
    """
    import nnsight
    from nnsight.intervention.batching import Batcher
    from nnsight.util import apply, applyn

    results = {}

    print("\n" + "=" * 60)
    print("PROFILING: Batching Operations")
    print("=" * 60)

    # Create test tensors of various sizes
    batch_sizes = [1, 4, 16, 64]
    seq_len = 128
    hidden_dim = 768  # Typical transformer hidden dimension

    # -------------------------------------------------------------
    # Profile 1: Tensor narrow() operation
    # -------------------------------------------------------------
    print("\n1. Tensor narrow() operation (used by Batcher.narrow)...")

    narrow_results = {}
    for batch_size in batch_sizes:
        tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        if batch_size > 1:
            # Narrow to get first half of batch
            def narrow_tensor(t=tensor, bs=batch_size):
                return t.narrow(0, 0, bs // 2)

            result = time_function(
                narrow_tensor,
                iterations=1000,
                warmup=100,
                name=f"narrow_batch_{batch_size}",
            )
            narrow_results[batch_size] = result
            print(f"   Batch size {batch_size}: {result}")

    results["tensor_narrow"] = narrow_results

    # -------------------------------------------------------------
    # Profile 2: Batcher.narrow() with nested structures
    # -------------------------------------------------------------
    print("\n2. Batcher.narrow() with nested structures...")

    batcher = Batcher()
    batcher.needs_batching = True

    # Simulate typical activation structure (tuple of tensors)
    def create_activation_structure(batch_size: int):
        return (
            torch.randn(batch_size, seq_len, hidden_dim, device=device),
            torch.randn(batch_size, seq_len, hidden_dim, device=device),
        )

    structure_results = {}
    for batch_size in batch_sizes:
        if batch_size < 4:
            continue

        structure = create_activation_structure(batch_size)
        batcher.current_value = structure
        batcher.last_batch_group = [0, batch_size]
        batch_group = [0, batch_size // 2]

        def narrow_structure(bg=batch_group):
            return batcher.narrow(bg)

        result = time_function(
            narrow_structure,
            iterations=1000,
            warmup=100,
            name=f"batcher_narrow_batch_{batch_size}",
        )
        structure_results[batch_size] = result
        print(f"   Batch size {batch_size} (tuple of 2 tensors): {result}")

    results["batcher_narrow_structure"] = structure_results

    # -------------------------------------------------------------
    # Profile 3: Batcher.swap() operation
    # -------------------------------------------------------------
    print("\n3. Batcher.swap() operation...")

    swap_results = {}
    for batch_size in batch_sizes:
        if batch_size < 4:
            continue

        original = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        swap_value = torch.randn(batch_size // 2, seq_len, hidden_dim, device=device)

        batcher.current_value = original.clone()
        batcher.last_batch_group = [0, batch_size]
        batcher.needs_batching = True
        batch_group = [0, batch_size // 2]

        def swap_tensor(bg=batch_group, sv=swap_value):
            batcher.current_value = original.clone()  # Reset each time
            batcher.swap(bg, sv)

        result = time_function(
            swap_tensor,
            iterations=500,
            warmup=50,
            name=f"batcher_swap_batch_{batch_size}",
        )
        swap_results[batch_size] = result
        print(f"   Batch size {batch_size}: {result}")

    results["batcher_swap"] = swap_results

    # -------------------------------------------------------------
    # Profile 4: apply() utility function
    # -------------------------------------------------------------
    print("\n4. apply() utility function overhead...")

    # apply() traverses data structures and applies a function to tensors
    test_tensor = torch.randn(16, seq_len, hidden_dim, device=device)

    def identity(t):
        return t

    def apply_identity():
        return apply(test_tensor, identity, torch.Tensor)

    result_apply_single = time_function(
        apply_identity,
        iterations=10000,
        warmup=1000,
        name="apply_single_tensor",
    )
    results["apply_single_tensor"] = result_apply_single
    print(f"   Single tensor: {result_apply_single}")

    # Nested structure
    nested_structure = {
        "hidden_states": (
            torch.randn(16, seq_len, hidden_dim, device=device),
            torch.randn(16, seq_len, hidden_dim, device=device),
        ),
        "attention_mask": torch.randn(16, seq_len, device=device),
    }

    def apply_nested():
        return apply(nested_structure, identity, torch.Tensor)

    result_apply_nested = time_function(
        apply_nested,
        iterations=5000,
        warmup=500,
        name="apply_nested_structure",
    )
    results["apply_nested_structure"] = result_apply_nested
    print(f"   Nested structure (3 tensors): {result_apply_nested}")

    # -------------------------------------------------------------
    # Profile 5: applyn() for multiple structures
    # -------------------------------------------------------------
    print("\n5. applyn() for multiple structures...")

    tensors_a = [torch.randn(16, seq_len, hidden_dim, device=device) for _ in range(3)]
    tensors_b = [torch.randn(16, seq_len, hidden_dim, device=device) for _ in range(3)]

    def add_tensors(*tensors):
        return sum(tensors)

    def applyn_add():
        return applyn([tensors_a, tensors_b], add_tensors, torch.Tensor)

    result_applyn = time_function(
        applyn_add,
        iterations=1000,
        warmup=100,
        name="applyn_two_lists",
    )
    results["applyn_two_lists"] = result_applyn
    print(f"   Two lists of 3 tensors: {result_applyn}")

    # -------------------------------------------------------------
    # Profile 6: Full multi-invoke batching
    # -------------------------------------------------------------
    print("\n6. Full multi-invoke batching overhead...")

    # Create a model and test with varying invoke counts
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 10)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    model = SimpleModel().to(device)
    wrapper = nnsight.NNsight(model)

    invoke_results = {}
    for num_invokes in [1, 2, 4, 8]:
        inputs = [torch.randn(1, hidden_dim, device=device) for _ in range(num_invokes)]

        def trace_with_invokes(n=num_invokes, ins=inputs):
            with wrapper.trace() as tracer:
                for i in range(n):
                    with tracer.invoke(ins[i]):
                        _ = wrapper.linear1.output.save()

        result = time_function(
            trace_with_invokes,
            iterations=20,
            warmup=5,
            name=f"multi_invoke_{num_invokes}",
        )
        invoke_results[num_invokes] = result
        print(f"   {num_invokes} invokes: {result}")

    results["multi_invoke_batching"] = invoke_results

    # Calculate per-invoke overhead
    if len(invoke_results) >= 2:
        base_time = invoke_results[1].mean_time_ms
        times = [(n, r.mean_time_ms) for n, r in invoke_results.items()]
        per_invoke_overhead = (times[-1][1] - times[0][1]) / (
            times[-1][0] - times[0][0]
        )
        results["per_invoke_overhead_ms"] = per_invoke_overhead
        print(f"\n   Per additional invoke overhead: {per_invoke_overhead:.3f}ms")

    # -------------------------------------------------------------
    # Profile 7: Tensor concatenation (used in batching)
    # -------------------------------------------------------------
    print("\n7. Tensor concatenation overhead...")

    concat_results = {}
    for num_tensors in [2, 4, 8]:
        tensors = [
            torch.randn(4, seq_len, hidden_dim, device=device)
            for _ in range(num_tensors)
        ]

        def concat_tensors(ts=tensors):
            return torch.cat(ts, dim=0)

        result = time_function(
            concat_tensors,
            iterations=500,
            warmup=50,
            name=f"concat_{num_tensors}_tensors",
        )
        concat_results[num_tensors] = result
        print(f"   Concatenate {num_tensors} tensors: {result}")

    results["tensor_concat"] = concat_results

    # -------------------------------------------------------------
    # Profile 8: In-place vs copy operations
    # -------------------------------------------------------------
    print("\n8. In-place vs copy operations...")

    test_tensor = torch.randn(16, seq_len, hidden_dim, device=device)
    update_tensor = torch.randn(4, seq_len, hidden_dim, device=device)

    # In-place slice assignment
    def inplace_assign():
        t = test_tensor.clone()
        t[0:4] = update_tensor
        return t

    result_inplace = time_function(
        inplace_assign,
        iterations=500,
        warmup=50,
        name="inplace_slice_assign",
    )
    results["inplace_slice_assign"] = result_inplace
    print(f"   In-place slice assign: {result_inplace}")

    # torch.cat approach
    def cat_assign():
        t = test_tensor.clone()
        pre = t.narrow(0, 4, 12)  # Rest of batch
        return torch.cat([update_tensor, pre], dim=0)

    result_cat = time_function(
        cat_assign,
        iterations=500,
        warmup=50,
        name="cat_based_assign",
    )
    results["cat_based_assign"] = result_cat
    print(f"   torch.cat based assign: {result_cat}")

    print(
        f"\n   In-place is {result_cat.mean_time_ms / result_inplace.mean_time_ms:.1f}x faster"
    )

    # -------------------------------------------------------------
    # Summary analysis
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("BATCHING OPERATIONS SUMMARY")
    print("-" * 60)

    print(f"\nPer-operation overhead:")
    if narrow_results:
        avg_narrow = sum(r.mean_time_ms for r in narrow_results.values()) / len(
            narrow_results
        )
        print(f"  - tensor.narrow():     {avg_narrow:.6f}ms")

    print(f"  - apply() single:      {result_apply_single.mean_time_ms:.6f}ms")
    print(f"  - apply() nested:      {result_apply_nested.mean_time_ms:.6f}ms")
    print(f"  - applyn() two lists:  {result_applyn.mean_time_ms:.6f}ms")

    if swap_results:
        avg_swap = sum(r.mean_time_ms for r in swap_results.values()) / len(
            swap_results
        )
        print(f"  - batcher.swap():      {avg_swap:.4f}ms")

    print(f"\nMulti-invoke scaling:")
    for n, result in invoke_results.items():
        print(f"  {n} invoke(s): {result.mean_time_ms:.2f}ms")

    return results


def get_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations based on profiling results."""
    recommendations = []

    # Per-invoke overhead
    if "per_invoke_overhead_ms" in results:
        overhead = results["per_invoke_overhead_ms"]
        if overhead > 1.0:
            recommendations.append(
                f"⚠ Each additional invoke adds ~{overhead:.2f}ms overhead. "
                f"Consider combining interventions into fewer invokes when possible."
            )
        else:
            recommendations.append(
                f"✓ Per-invoke overhead is acceptable ({overhead:.2f}ms)."
            )

    # apply() overhead
    if "apply_nested_structure" in results:
        apply_time = results["apply_nested_structure"].mean_time_ms
        if apply_time > 0.1:
            recommendations.append(
                f"⚠ apply() on nested structures takes {apply_time:.4f}ms. "
                f"Consider caching or avoiding deep nesting where possible."
            )

    # In-place vs concat
    if "inplace_slice_assign" in results and "cat_based_assign" in results:
        inplace = results["inplace_slice_assign"].mean_time_ms
        cat = results["cat_based_assign"].mean_time_ms
        if cat > inplace * 2:
            recommendations.append(
                f"⚠ torch.cat-based swap is {cat/inplace:.1f}x slower than in-place. "
                f"Consider optimizing Batcher.swap() to prefer in-place when safe."
            )

    # Swap overhead scaling
    if "batcher_swap" in results:
        swap_results = results["batcher_swap"]
        if len(swap_results) >= 2:
            sizes = sorted(swap_results.keys())
            times = [swap_results[s].mean_time_ms for s in sizes]
            scaling = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]

            if scaling > size_ratio * 1.5:
                recommendations.append(
                    f"⚠ Batcher.swap() scales worse than linearly with batch size. "
                    f"({sizes[0]}→{sizes[-1]}: {scaling:.1f}x time for {size_ratio}x size)"
                )

    return recommendations


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = run_batching_profiles(device)

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    for rec in get_recommendations(results):
        print(f"\n{rec}")
