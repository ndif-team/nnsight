#!/usr/bin/env python3
"""
Comprehensive NNsight Performance Profiling

This script runs all profilers and generates a detailed performance report
with actionable recommendations for optimization.

Usage:
    python run_full_profile.py [--device cuda|cpu] [--output report.md]

Based on NNsight's architecture as documented in NNsight.md:
- Section 2: Tracing (capture, parse, compile)
- Section 3: Interleaving (threading, mediators, events)
- Section 4: Envoy (module wrapping, attribute access)
- Batching: narrow/swap operations for multi-invoke
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch


def import_profilers():
    """Import all profiler modules."""
    # Add the current directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from profile_tracing import (
        run_tracing_profiles,
        get_recommendations as tracing_recs,
    )
    from profile_interleaving import (
        run_interleaving_profiles,
        get_recommendations as interleaving_recs,
    )
    from profile_envoy import run_envoy_profiles, get_recommendations as envoy_recs
    from profile_batching import (
        run_batching_profiles,
        get_recommendations as batching_recs,
    )

    return {
        "tracing": (run_tracing_profiles, tracing_recs),
        "interleaving": (run_interleaving_profiles, interleaving_recs),
        "envoy": (run_envoy_profiles, envoy_recs),
        "batching": (run_batching_profiles, batching_recs),
    }


def run_all_profiles(device: str, skip_profilers: list = None) -> Dict[str, Any]:
    """Run all profilers and collect results."""
    import sys

    profilers = import_profilers()
    all_results = {}
    all_recommendations = {}

    if skip_profilers is None:
        skip_profilers = []

    for name, (run_fn, rec_fn) in profilers.items():
        if name in skip_profilers:
            print(f"\n[SKIPPED] {name.upper()} profiler")
            continue

        print(f"\n{'#'*60}", flush=True)
        print(f"# Running {name.upper()} profiler", flush=True)
        print(f"{'#'*60}", flush=True)
        sys.stdout.flush()

        try:
            results = run_fn(device)
            recommendations = rec_fn(results)

            all_results[name] = results
            all_recommendations[name] = recommendations
            print(f"\n[DONE] {name.upper()} profiler completed", flush=True)

        except Exception as e:
            print(f"ERROR in {name} profiler: {e}", flush=True)
            import traceback

            traceback.print_exc()
            all_results[name] = {"error": str(e)}
            all_recommendations[name] = [f"⚠ Profiler failed: {e}"]

    return all_results, all_recommendations


def generate_report(
    results: Dict[str, Any],
    recommendations: Dict[str, List[str]],
    device: str,
) -> str:
    """Generate a comprehensive markdown report."""

    lines = []

    # Header
    lines.append("# NNsight Performance Profiling Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Device:** {device}")
    if torch.cuda.is_available():
        lines.append(f"**GPU:** {torch.cuda.get_device_name()}")
    lines.append(f"**PyTorch Version:** {torch.__version__}")

    try:
        import nnsight

        lines.append(
            f"**NNsight Version:** {getattr(nnsight, '__version__', 'unknown')}"
        )
    except:
        pass

    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report profiles the major performance-critical components of NNsight:"
    )
    lines.append("")
    lines.append("1. **Tracing Phase** - Source capture, AST parsing, code compilation")
    lines.append(
        "2. **Interleaving Phase** - Threading, event queues, cross-thread communication"
    )
    lines.append(
        "3. **Envoy System** - Module wrapping, attribute access, .output/.input handling"
    )
    lines.append("4. **Batching Operations** - narrow/swap for multi-invoke scenarios")
    lines.append("")

    # High-level metrics
    lines.append("### Key Performance Metrics")
    lines.append("")

    key_metrics = []

    # Extract key metrics from results
    if "tracing" in results and not "error" in results["tracing"]:
        tracing = results["tracing"]
        if "full_trace_cached" in tracing:
            key_metrics.append(
                f"- **Cached trace overhead:** {tracing['full_trace_cached'].mean_time_ms:.2f}ms"
            )
        if "cache_improvement_ratio" in tracing:
            key_metrics.append(
                f"- **Trace caching speedup:** {tracing['cache_improvement_ratio']:.1f}x"
            )

    if "interleaving" in results and not "error" in results["interleaving"]:
        interleaving = results["interleaving"]
        if "cross_thread_roundtrip_ms" in interleaving:
            key_metrics.append(
                f"- **Cross-thread roundtrip:** {interleaving['cross_thread_roundtrip_ms']:.4f}ms"
            )
        if "cross_invoker_overhead_ms" in interleaving:
            key_metrics.append(
                f"- **CROSS_INVOKER overhead:** {interleaving['cross_invoker_overhead_ms']:.2f}ms"
            )

    if "envoy" in results and not "error" in results["envoy"]:
        envoy = results["envoy"]
        if "per_intervention_overhead_ms" in envoy:
            key_metrics.append(
                f"- **Per-intervention overhead:** {envoy['per_intervention_overhead_ms']:.2f}ms"
            )

    if "batching" in results and not "error" in results["batching"]:
        batching = results["batching"]
        if "per_invoke_overhead_ms" in batching:
            key_metrics.append(
                f"- **Per-invoke batching overhead:** {batching['per_invoke_overhead_ms']:.2f}ms"
            )

    for metric in key_metrics:
        lines.append(metric)

    lines.append("")

    # Recommendations Section
    lines.append("## Optimization Recommendations")
    lines.append("")

    all_recs = []
    for category, recs in recommendations.items():
        for rec in recs:
            all_recs.append((category, rec))

    # Separate into warnings and positive findings
    warnings = [(c, r) for c, r in all_recs if r.startswith("⚠")]
    positives = [(c, r) for c, r in all_recs if r.startswith("✓")]

    if warnings:
        lines.append("### Areas for Improvement")
        lines.append("")
        for category, rec in warnings:
            lines.append(f"**{category.title()}:** {rec[2:]}")  # Remove emoji
            lines.append("")

    if positives:
        lines.append("### Well-Optimized Areas")
        lines.append("")
        for category, rec in positives:
            lines.append(f"- **{category.title()}:** {rec[2:]}")
        lines.append("")

    # Detailed Results
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Profiling Results")
    lines.append("")

    # Tracing Section
    if "tracing" in results and not "error" in results["tracing"]:
        lines.append("### 1. Tracing Phase")
        lines.append("")
        tracing = results["tracing"]

        lines.append("| Operation | Mean Time | Notes |")
        lines.append("|-----------|-----------|-------|")

        if "full_trace_uncached" in tracing:
            lines.append(
                f"| Full trace (uncached) | {tracing['full_trace_uncached'].mean_time_ms:.3f}ms | "
                f"Includes source capture, AST parsing, compilation |"
            )
        if "full_trace_cached" in tracing:
            lines.append(
                f"| Full trace (cached) | {tracing['full_trace_cached'].mean_time_ms:.3f}ms | "
                f"Skips source capture and parsing |"
            )
        if "ast_parsing" in tracing:
            lines.append(
                f"| AST parsing | {tracing['ast_parsing'].mean_time_ms:.4f}ms | "
                f"Parsing source code into AST |"
            )
        if "inspect_getsourcelines" in tracing:
            lines.append(
                f"| inspect.getsourcelines | {tracing['inspect_getsourcelines'].mean_time_ms:.4f}ms | "
                f"Extracting source from file |"
            )
        if "code_compilation" in tracing:
            lines.append(
                f"| Code compilation | {tracing['code_compilation'].mean_time_ms:.4f}ms | "
                f"compile() call |"
            )

        lines.append("")

        # Complexity scaling
        if "complexity_scaling" in tracing:
            lines.append("**Intervention Scaling:**")
            lines.append("")
            lines.append("| Interventions | Time (ms) |")
            lines.append("|---------------|-----------|")
            for n, result in tracing["complexity_scaling"].items():
                lines.append(f"| {n} | {result.mean_time_ms:.2f} |")
            lines.append("")

    # Interleaving Section
    if "interleaving" in results and not "error" in results["interleaving"]:
        lines.append("### 2. Interleaving Phase")
        lines.append("")
        interleaving = results["interleaving"]

        lines.append("| Operation | Mean Time | Notes |")
        lines.append("|-----------|-----------|-------|")

        if "thread_start" in interleaving:
            lines.append(
                f"| Thread creation & start | {interleaving['thread_start'].mean_time_ms:.4f}ms | "
                f"Per mediator |"
            )
        if "lock_acquire_release" in interleaving:
            lines.append(
                f"| Lock acquire/release | {interleaving['lock_acquire_release'].mean_time_ms:.6f}ms | "
                f"Per event |"
            )
        if "cross_thread_roundtrip_ms" in interleaving:
            lines.append(
                f"| Cross-thread roundtrip | {interleaving['cross_thread_roundtrip_ms']:.4f}ms | "
                f"Main thread ↔ worker thread |"
            )
        if "hook_registration" in interleaving:
            lines.append(
                f"| Hook registration | {interleaving['hook_registration'].mean_time_ms:.4f}ms | "
                f"register_forward_hook + remove |"
            )

        lines.append("")

    # Envoy Section
    if "envoy" in results and not "error" in results["envoy"]:
        lines.append("### 3. Envoy System")
        lines.append("")
        envoy = results["envoy"]

        if "envoy_construction" in envoy:
            lines.append("**Envoy Tree Construction:**")
            lines.append("")
            lines.append("| Layers | Modules | Time (ms) | Per Module |")
            lines.append("|--------|---------|-----------|------------|")
            for n_layers, data in envoy["envoy_construction"].items():
                lines.append(
                    f"| {n_layers} | {data['num_modules']} | "
                    f"{data['timing'].mean_time_ms:.2f} | {data['time_per_module']:.4f}ms |"
                )
            lines.append("")

        lines.append("**Attribute Access:**")
        lines.append("")
        lines.append("| Access Pattern | Time |")
        lines.append("|----------------|------|")
        if "child_access" in envoy:
            lines.append(
                f"| Direct child (.layers) | {envoy['child_access'].mean_time_ms*1000:.2f}μs |"
            )
        if "deep_path_access" in envoy:
            lines.append(
                f"| Deep path (.layers[5].linear1) | {envoy['deep_path_access'].mean_time_ms*1000:.2f}μs |"
            )
        if "module_attribute" in envoy:
            lines.append(
                f"| Module attribute (.weight) | {envoy['module_attribute'].mean_time_ms*1000:.2f}μs |"
            )
        lines.append("")

    # Batching Section
    if "batching" in results and not "error" in results["batching"]:
        lines.append("### 4. Batching Operations")
        lines.append("")
        batching = results["batching"]

        if "multi_invoke_batching" in batching:
            lines.append("**Multi-Invoke Scaling:**")
            lines.append("")
            lines.append("| Invokes | Time (ms) |")
            lines.append("|---------|-----------|")
            for n, result in batching["multi_invoke_batching"].items():
                lines.append(f"| {n} | {result.mean_time_ms:.2f} |")
            lines.append("")

    # Architecture Insights
    lines.append("---")
    lines.append("")
    lines.append("## Architecture Analysis")
    lines.append("")
    lines.append(
        "Based on NNsight's design (see NNsight.md), the following architectural "
    )
    lines.append("decisions have significant performance implications:")
    lines.append("")

    lines.append("### 1. Threading Model")
    lines.append("")
    lines.append("NNsight uses a **ping-pong threading model** where:")
    lines.append("- Main thread runs model forward pass")
    lines.append("- Worker threads run intervention code")
    lines.append("- Communication via lock-based queues")
    lines.append("")
    lines.append(
        "**Impact:** Each `.output` access requires a cross-thread roundtrip. "
    )
    lines.append("Reducing intervention count directly improves performance.")
    lines.append("")

    lines.append("### 2. Source Tracing")
    lines.append("")
    lines.append("NNsight captures source code at runtime:")
    lines.append("- Walks call stack to find source")
    lines.append("- Parses AST to extract with-block")
    lines.append("- Compiles into callable function")
    lines.append("")
    lines.append("**Impact:** `TRACE_CACHING=True` (default) caches this work. ")
    lines.append("First execution is slower; subsequent executions are faster.")
    lines.append("")

    lines.append("### 3. Eager Envoy Construction")
    lines.append("")
    lines.append("When `NNsight(model)` is called:")
    lines.append("- Creates Envoy for every module")
    lines.append("- Installs hooks on every module")
    lines.append("- Builds complete module tree")
    lines.append("")
    lines.append("**Impact:** Initialization cost scales with model size. ")
    lines.append(
        "For very large models, consider using `dispatch=True` for lazy loading."
    )
    lines.append("")

    lines.append("### 4. Hook-Based Interception")
    lines.append("")
    lines.append("Every module gets input and output hooks:")
    lines.append("- `register_forward_pre_hook` for inputs")
    lines.append("- `register_forward_hook` for outputs")
    lines.append("")
    lines.append("**Impact:** Hooks fire for every module even if not accessed. ")
    lines.append(
        "The interleaving flag check (`if not self.interleaving: return`) is optimized "
    )
    lines.append("but still adds minimal overhead per module forward pass.")
    lines.append("")

    # Potential Optimizations
    lines.append("---")
    lines.append("")
    lines.append("## Potential Optimizations")
    lines.append("")
    lines.append("### Without Breaking Architecture")
    lines.append("")
    lines.append("1. **Thread Pooling for Mediators**")
    lines.append("   - Currently creates new thread per invoke")
    lines.append("   - Could reuse threads from a pool")
    lines.append("   - Saves ~0.1-0.5ms per invoke")
    lines.append("")
    lines.append("2. **Lazy Hook Registration**")
    lines.append("   - Only register hooks on modules that will be accessed")
    lines.append("   - Requires static analysis or two-pass approach")
    lines.append("   - Would reduce overhead for large models with few interventions")
    lines.append("")
    lines.append("3. **Batch String Operations**")
    lines.append("   - Provider strings built on every hook call")
    lines.append("   - Could cache path → provider mappings")
    lines.append("")
    lines.append("4. **Optimize Batcher.swap()**")
    lines.append("   - Currently uses torch.cat for some cases")
    lines.append("   - In-place operations are faster when safe")
    lines.append("")
    lines.append("5. **Pre-compile Common Patterns**")
    lines.append("   - Cache compiled intervention functions")
    lines.append("   - Already partially done with TRACE_CACHING")
    lines.append("")

    lines.append("### User-Level Optimizations")
    lines.append("")
    lines.append("1. **Enable TRACE_CACHING** (default)")
    lines.append("   ```python")
    lines.append("   from nnsight import CONFIG")
    lines.append("   CONFIG.APP.TRACE_CACHING = True")
    lines.append("   ```")
    lines.append("")
    lines.append("2. **Disable CROSS_INVOKER when not needed**")
    lines.append("   ```python")
    lines.append("   CONFIG.APP.CROSS_INVOKER = False")
    lines.append("   ```")
    lines.append("")
    lines.append("3. **Minimize intervention count**")
    lines.append("   - Combine related operations in single save")
    lines.append(
        "   - Use list saves: `outputs = [layer.output for layer in layers].save()`"
    )
    lines.append("")
    lines.append("4. **Use dispatch=True for large models**")
    lines.append("   - Defers weight loading until needed")
    lines.append("   - Reduces initial memory footprint")
    lines.append("")

    return "\n".join(lines)


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save raw results as JSON for further analysis."""

    def serialize(obj):
        """Convert non-serializable objects to dicts."""
        if hasattr(obj, "__dict__"):
            return {
                k: serialize(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serialized = serialize(results)

    output_file = output_dir / "profiling_results.json"
    with open(output_file, "w") as f:
        json.dump(serialized, f, indent=2)

    print(f"\nRaw results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="NNsight comprehensive performance profiling"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run profiling on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="performance_report.md",
        help="Output file for the report",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NNsight Comprehensive Performance Profiling")
    print("=" * 60)
    print(f"\nDevice: {args.device}")
    print(f"Output: {output_dir / args.output}")
    print()

    # Run all profiles
    start_time = time.time()
    results, recommendations = run_all_profiles(args.device)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Profiling completed in {total_time:.1f} seconds")
    print(f"{'='*60}")

    # Generate report
    report = generate_report(results, recommendations, args.device)

    # Save report
    report_file = output_dir / args.output
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Save raw results
    save_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("TOP RECOMMENDATIONS")
    print("=" * 60)

    warnings = []
    for category, recs in recommendations.items():
        for rec in recs:
            if rec.startswith("⚠"):
                warnings.append((category, rec))

    for category, rec in warnings[:5]:
        print(f"\n[{category.upper()}] {rec}")


if __name__ == "__main__":
    main()
