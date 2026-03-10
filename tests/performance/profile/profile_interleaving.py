"""
Profile the interleaving phase of NNsight.

This module profiles:
1. Thread creation and startup overhead
2. Event queue communication (lock acquire/release)
3. Provider/requester matching overhead
4. Iteration tracking overhead
5. Cross-invoker variable sharing

These correspond to NNsight's interleaving process as documented
in NNsight.md Section 3.
"""

import gc
import time
import threading
import _thread
from typing import Dict, List, Any
from collections import defaultdict

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


def create_deep_model(num_layers: int = 12, hidden_dim: int = 64) -> nn.Module:
    """Create a deep model for testing."""

    class DeepModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final = nn.Linear(hidden_dim, 10)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.final(x)

    return DeepModel()


def run_interleaving_profiles(device: str = "cuda") -> Dict[str, Any]:
    """
    Run comprehensive profiling of NNsight's interleaving phase.

    Returns:
        Dictionary with profiling results
    """
    import nnsight
    from nnsight import CONFIG
    from nnsight.intervention.interleaver import Mediator

    results = {}
    timer = InstrumentedTimer()

    print("\n" + "=" * 60)
    print("PROFILING: Interleaving Phase")
    print("=" * 60)

    # Create model
    model = create_deep_model(num_layers=12).to(device)
    wrapper = nnsight.NNsight(model)
    input_tensor = torch.randn(1, 64, device=device)

    # -------------------------------------------------------------
    # Profile 1: Thread creation overhead
    # -------------------------------------------------------------
    print("\n1. Thread creation overhead...")

    def create_thread():
        def dummy_worker():
            pass

        t = threading.Thread(target=dummy_worker, daemon=True)
        return t

    result_thread_create = time_function(
        create_thread,
        iterations=100,
        warmup=10,
        name="thread_creation",
    )
    results["thread_creation"] = result_thread_create
    print(f"   {result_thread_create}")

    # Thread start overhead
    def create_and_start_thread():
        event = threading.Event()

        def worker():
            event.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        event.wait()  # Ensure thread actually runs

    result_thread_start = time_function(
        create_and_start_thread,
        iterations=50,
        warmup=5,
        name="thread_start",
    )
    results["thread_start"] = result_thread_start
    print(f"   {result_thread_start}")

    # -------------------------------------------------------------
    # Profile 2: Lock-based queue overhead (as used in Mediator.Value)
    # -------------------------------------------------------------
    print("\n2. Lock-based queue overhead...")

    # Profile the exact mechanism used by Mediator.Value
    class MediatorValue:
        def __init__(self):
            self.value = None
            self.lock = _thread.allocate_lock()
            self.lock.acquire()
            self.has_value = False

        def get(self):
            value = self.value
            self.value = None
            self.has_value = False
            return value

        def wait(self):
            self.lock.acquire()

        def put(self, value):
            self.value = value
            self.has_value = True
            self.lock.release()

    # Single lock acquire/release
    def lock_cycle():
        lock = _thread.allocate_lock()
        lock.acquire()
        lock.release()

    result_lock = time_function(
        lock_cycle,
        iterations=1000,
        warmup=100,
        name="lock_acquire_release",
    )
    results["lock_acquire_release"] = result_lock
    print(f"   Lock acquire/release: {result_lock}")

    # Full event queue round-trip (simulating ping-pong)
    def queue_roundtrip():
        val = MediatorValue()
        val.put("test")
        _ = val.get()

    result_queue = time_function(
        queue_roundtrip,
        iterations=1000,
        warmup=100,
        name="queue_roundtrip",
    )
    results["queue_roundtrip"] = result_queue
    print(f"   Queue round-trip: {result_queue}")

    # Cross-thread communication overhead
    roundtrip_times = []

    def ping_pong_worker(request_queue, response_queue, iterations):
        for _ in range(iterations):
            request_queue.wait()
            data = request_queue.get()
            response_queue.put(data + 1)

    def cross_thread_roundtrip(n_roundtrips=100):
        request_q = MediatorValue()
        response_q = MediatorValue()

        worker = threading.Thread(
            target=ping_pong_worker,
            args=(request_q, response_q, n_roundtrips),
            daemon=True,
        )
        worker.start()

        for i in range(n_roundtrips):
            request_q.put(i)
            response_q.wait()
            _ = response_q.get()

        worker.join(timeout=1.0)

    sync_cuda()
    start = time.perf_counter()
    cross_thread_roundtrip(100)
    sync_cuda()
    elapsed = (time.perf_counter() - start) * 1000 / 100  # per roundtrip

    results["cross_thread_roundtrip_ms"] = elapsed
    print(f"   Cross-thread round-trip: {elapsed:.4f}ms per round-trip")

    # -------------------------------------------------------------
    # Profile 3: Provider string operations
    # -------------------------------------------------------------
    print("\n3. Provider string operations...")

    # String concatenation for iteration suffix
    def iterate_provider_string():
        provider = "model.transformer.h.0.output"
        iteration = 5
        return f"{provider}.i{iteration}"

    result_str_concat = time_function(
        iterate_provider_string,
        iterations=10000,
        warmup=1000,
        name="provider_string_concat",
    )
    results["provider_string_concat"] = result_str_concat
    print(f"   String concatenation: {result_str_concat}")

    # Dictionary lookup for iteration tracking
    iteration_tracker = defaultdict(int)
    providers = [f"model.transformer.h.{i}.output" for i in range(12)]

    def iteration_tracking():
        for p in providers:
            iteration_tracker[p] += 1

    result_tracking = time_function(
        iteration_tracking,
        iterations=1000,
        warmup=100,
        name="iteration_tracking",
    )
    results["iteration_tracking"] = result_tracking
    print(f"   Iteration tracking (12 modules): {result_tracking}")

    # Set membership (history check)
    history = set(providers[:6])

    def history_check():
        for p in providers:
            _ = p in history

    result_history = time_function(
        history_check,
        iterations=10000,
        warmup=1000,
        name="history_set_check",
    )
    results["history_set_check"] = result_history
    print(f"   History set check: {result_history}")

    # -------------------------------------------------------------
    # Profile 4: Hook registration overhead
    # -------------------------------------------------------------
    print("\n4. Hook registration overhead...")

    test_module = nn.Linear(64, 64).to(device)

    def hook_fn(module, args, output):
        return output

    def register_and_remove_hook():
        handle = test_module.register_forward_hook(hook_fn)
        handle.remove()

    result_hook_reg = time_function(
        register_and_remove_hook,
        iterations=1000,
        warmup=100,
        name="hook_registration",
    )
    results["hook_registration"] = result_hook_reg
    print(f"   Register/remove hook: {result_hook_reg}")

    # Multiple hooks
    def register_multiple_hooks(n=10):
        handles = []
        for _ in range(n):
            h1 = test_module.register_forward_pre_hook(
                lambda m, a: a, with_kwargs=False
            )
            h2 = test_module.register_forward_hook(hook_fn)
            handles.extend([h1, h2])
        for h in handles:
            h.remove()

    result_multi_hook = time_function(
        lambda: register_multiple_hooks(10),
        iterations=100,
        warmup=10,
        name="multiple_hook_registration",
    )
    results["multiple_hook_registration"] = result_multi_hook
    print(f"   Register/remove 20 hooks: {result_multi_hook}")

    # -------------------------------------------------------------
    # Profile 5: Cross-invoker variable sharing overhead
    # -------------------------------------------------------------
    print("\n5. Cross-invoker variable sharing...")

    # Simulate push/pull operations
    import ctypes

    def simulate_push_variables():
        # This simulates what push_variables does
        frame_locals = {"var1": torch.randn(10), "var2": "test", "var3": 42}
        target_locals = {}
        target_locals.update(frame_locals)

    result_push = time_function(
        simulate_push_variables,
        iterations=1000,
        warmup=100,
        name="variable_push",
    )
    results["variable_push"] = result_push
    print(f"   Variable push (3 vars): {result_push}")

    # Globals update (as used in intervention function)
    test_globals = {"x": 1, "y": 2}
    updates = {"z": 3, "w": 4}

    def globals_update():
        test_globals.update(updates)

    result_globals = time_function(
        globals_update,
        iterations=10000,
        warmup=1000,
        name="globals_update",
    )
    results["globals_update"] = result_globals
    print(f"   Globals update: {result_globals}")

    # -------------------------------------------------------------
    # Profile 6: Full interleaving with varying mediator counts
    # -------------------------------------------------------------
    print("\n6. Full interleaving with varying mediator counts...")

    original_cross_invoker = CONFIG.APP.CROSS_INVOKER

    mediator_results = {}
    for num_invokes in [1, 2, 4]:

        def trace_with_invokes(n=num_invokes):
            with wrapper.trace() as tracer:
                for i in range(n):
                    with tracer.invoke(input_tensor):
                        _ = wrapper.layers[0].output.save()

        result = time_function(
            trace_with_invokes,
            iterations=10,
            warmup=2,
            name=f"interleaving_{num_invokes}_mediators",
        )
        mediator_results[num_invokes] = result
        print(f"   {num_invokes} mediator(s): {result.mean_time_ms:.3f}ms")

    results["mediator_scaling"] = mediator_results

    # Test with cross-invoker disabled
    print("\n7. Cross-invoker disabled comparison...")

    CONFIG.APP.CROSS_INVOKER = False

    def trace_no_cross_invoker():
        with wrapper.trace() as tracer:
            with tracer.invoke(input_tensor):
                _ = wrapper.layers[0].output.save()
            with tracer.invoke(input_tensor):
                _ = wrapper.layers[1].output.save()

    result_no_cross = time_function(
        trace_no_cross_invoker,
        iterations=10,
        warmup=2,
        name="no_cross_invoker",
    )
    results["no_cross_invoker"] = result_no_cross
    print(f"   CROSS_INVOKER=False: {result_no_cross}")

    CONFIG.APP.CROSS_INVOKER = True

    def trace_with_cross_invoker():
        with wrapper.trace() as tracer:
            with tracer.invoke(input_tensor):
                _ = wrapper.layers[0].output.save()
            with tracer.invoke(input_tensor):
                _ = wrapper.layers[1].output.save()

    result_with_cross = time_function(
        trace_with_cross_invoker,
        iterations=10,
        warmup=2,
        name="with_cross_invoker",
    )
    results["with_cross_invoker"] = result_with_cross
    print(f"   CROSS_INVOKER=True:  {result_with_cross}")

    CONFIG.APP.CROSS_INVOKER = original_cross_invoker

    cross_invoker_overhead = (
        result_with_cross.mean_time_ms - result_no_cross.mean_time_ms
    )
    results["cross_invoker_overhead_ms"] = cross_invoker_overhead
    print(f"   Cross-invoker overhead: {cross_invoker_overhead:.3f}ms")

    # -------------------------------------------------------------
    # Summary analysis
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("INTERLEAVING PHASE SUMMARY")
    print("-" * 60)

    print(f"\nPer-operation overhead breakdown:")
    print(f"  - Thread start:           {result_thread_start.mean_time_ms:.4f}ms")
    print(f"  - Lock cycle:             {result_lock.mean_time_ms:.6f}ms")
    print(f"  - Cross-thread roundtrip: {results['cross_thread_roundtrip_ms']:.4f}ms")
    print(f"  - Provider string ops:    {result_str_concat.mean_time_ms:.6f}ms")
    print(f"  - Hook register/remove:   {result_hook_reg.mean_time_ms:.4f}ms")

    # Estimate overhead per intervention
    avg_single_trace = mediator_results[1].mean_time_ms
    num_modules = 12  # layers in our test model

    print(f"\nEstimated overhead per intervention:")
    print(f"  - Base trace overhead: ~{avg_single_trace:.2f}ms")
    print(
        f"  - Per cross-thread roundtrip: ~{results['cross_thread_roundtrip_ms']:.4f}ms"
    )

    # Scaling analysis
    if len(mediator_results) >= 2:
        t1 = mediator_results[1].mean_time_ms
        t2 = mediator_results[2].mean_time_ms
        per_mediator = t2 - t1
        print(f"  - Per additional mediator: ~{per_mediator:.2f}ms")

    return results


def get_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations based on profiling results."""
    recommendations = []

    # Thread creation overhead
    if "thread_start" in results:
        thread_time = results["thread_start"].mean_time_ms
        if thread_time > 0.5:
            recommendations.append(
                f"⚠ Thread creation takes {thread_time:.2f}ms per mediator. "
                f"Consider thread pooling for multiple invokes."
            )

    # Cross-thread roundtrip
    if "cross_thread_roundtrip_ms" in results:
        roundtrip = results["cross_thread_roundtrip_ms"]
        if roundtrip > 0.05:
            recommendations.append(
                f"⚠ Cross-thread roundtrip takes {roundtrip:.4f}ms. "
                f"This is fundamental to the ping-pong execution model. "
                f"Reducing intervention count directly reduces this overhead."
            )
        else:
            recommendations.append(
                f"✓ Cross-thread communication is efficient ({roundtrip:.4f}ms per round-trip)."
            )

    # Cross-invoker overhead
    if "cross_invoker_overhead_ms" in results:
        overhead = results["cross_invoker_overhead_ms"]
        if overhead > 1.0:
            recommendations.append(
                f"⚠ CROSS_INVOKER adds {overhead:.2f}ms overhead. "
                f"Disable with CONFIG.APP.CROSS_INVOKER = False when not needed."
            )
        else:
            recommendations.append(
                f"✓ CROSS_INVOKER overhead is minimal ({overhead:.2f}ms)."
            )

    # Hook registration
    if "hook_registration" in results:
        hook_time = results["hook_registration"].mean_time_ms
        if hook_time > 0.05:
            recommendations.append(
                f"⚠ Hook registration takes {hook_time:.4f}ms. "
                f"Consider hook pooling or lazy hook registration."
            )

    # Mediator scaling
    if "mediator_scaling" in results:
        scaling = results["mediator_scaling"]
        if len(scaling) >= 2:
            times = [(n, r.mean_time_ms) for n, r in scaling.items()]
            t1, t4 = times[0][1], times[-1][1]
            ratio = t4 / t1
            n1, n4 = times[0][0], times[-1][0]
            expected_ratio = n4 / n1

            if ratio > expected_ratio * 1.5:
                recommendations.append(
                    f"⚠ Mediator overhead scales superlinearly ({n1}→{n4}: {ratio:.1f}x vs expected {expected_ratio}x). "
                    f"Consider batching multiple interventions into single invokes when possible."
                )

    return recommendations


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = run_interleaving_profiles(device)

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    for rec in get_recommendations(results):
        print(f"\n{rec}")
