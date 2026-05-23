"""Profile PP pull overhead vs in-stage intervention.

Measures:
  A. Pull latency vs tensor size
  B. Sequential pull scaling (1 to 50 pulls)
  C. Trace overhead comparison (bare, 1 save, 6 saves, multi-token)

Requires 2 GPUs.
Run: CUDA_VISIBLE_DEVICES=0,1 python tests/profile_pp_pull.py
"""

if __name__ == '__main__':
    import pickle
    import time

    from nnsight.modeling.vllm import VLLM

    model = VLLM(
        "openai-community/gpt2",
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.1,
        dispatch=True,
    )
    rpc = model.vllm_entrypoint.collective_rpc

    def profile_pull(num_pulls, shape, dtype_str, direction="0to1"):
        results = rpc("test_pp_profile_pull", args=(num_pulls, shape, dtype_str, direction))
        rpc("test_pp_buffer_clear")  # clean up after each run
        for r in results:
            info = pickle.loads(r)
            if info.get("role") == "puller":
                return info
        return None

    print("=" * 70)
    print("PP Pull Profiling — GPT-2 PP=2")
    print("=" * 70)

    # ==================================================================
    # Part A: Pull latency vs tensor size
    # ==================================================================
    print("\n--- Part A: Pull latency vs tensor size (5 pulls each, drop first) ---")
    print(f"{'Shape':<30} {'Bytes':>10} {'Mean (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 75)

    shapes = [
        ([1],               "bfloat16"),
        ([768],             "bfloat16"),
        ([1, 128, 768],     "bfloat16"),
        ([1, 512, 768],     "bfloat16"),
        ([1, 1024, 768],    "bfloat16"),
        ([1, 2048, 768],    "bfloat16"),
        ([1, 4096, 768],    "bfloat16"),
        ([1, 1024, 4096],   "bfloat16"),
    ]

    # Warmup: one pull to establish gloo TCP connection
    profile_pull(1, [768], "bfloat16")

    for shape, dt in shapes:
        info = profile_pull(5, shape, dt)
        if info:
            # Drop first pull (cold start per shape)
            times = info["times_ms"][1:]
            mean = sum(times) / len(times) if times else info["mean_ms"]
            mn = min(times) if times else info["min_ms"]
            mx = max(times) if times else info["max_ms"]
            print(f"{str(shape):<30} {info['tensor_bytes']:>10,} {mean:>10.2f} {mn:>10.2f} {mx:>10.2f}")

    # ==================================================================
    # Part B: Sequential pull scaling
    # ==================================================================
    print("\n--- Part B: Sequential pull scaling ([1, 512, 768] = 768KB) ---")
    print(f"{'N pulls':>10} {'Total (ms)':>12} {'Mean (ms)':>12} {'Throughput':>15}")
    print("-" * 55)

    hidden_shape = [1, 512, 768]
    for n in [1, 2, 5, 10, 20, 50]:
        info = profile_pull(n, hidden_shape, "bfloat16")
        if info:
            tp = info['tensor_bytes'] * n / (info['total_ms'] / 1000) / 1e9
            print(f"{n:>10} {info['total_ms']:>12.2f} {info['mean_ms']:>12.2f} {tp:>12.2f} GB/s")

    # ==================================================================
    # Part C: Trace overhead comparison
    # ==================================================================
    print("\n--- Part C: Trace overhead ---")

    def bench_trace(name, fn, warmup=3, iters=10):
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        mean = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        print(f"{name:<45} {mean:>8.2f} ms  (min={mn:.2f}, max={mx:.2f})")
        return mean

    def bare_trace():
        with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
            model.logits.output.save()

    def trace_1save():
        with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
            model.transformer.h[0].output[0].save()
            model.logits.output.save()

    def trace_6save():
        with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
            for i in range(6):
                model.transformer.h[i].output[0].save()
            model.logits.output.save()

    def trace_multigen():
        with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1, max_tokens=3) as tracer:
            logit_list = list().save()
            for step in tracer.iter[0:3]:
                logit_list.append(model.logits.output)

    t_bare = bench_trace("Bare trace (logits only)", bare_trace)
    t_1save = bench_trace("Trace + 1 in-stage save", trace_1save)
    t_6save = bench_trace("Trace + 6 in-stage saves", trace_6save)

    try:
        t_gen = bench_trace("Multi-token gen (3 steps)", trace_multigen, warmup=2, iters=5)
    except Exception as e:
        print(f"{'Multi-token gen (3 steps)':<45} FAILED: {e}")
        t_gen = None

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n--- Summary ---")
    print(f"Forward pass (bare trace):           {t_bare:.2f} ms")
    per_save = (t_6save - t_bare) / 6
    print(f"Per in-stage save overhead:           {per_save:.2f} ms")

    info = profile_pull(10, [1, 512, 768], "bfloat16")
    if info:
        pull_ms = info["mean_ms"]
        print(f"Per cross-stage pull (768KB bf16):   {pull_ms:.2f} ms")
        print(f"Pull / forward ratio:                {pull_ms / t_bare * 100:.1f}%")
        affordable = int(t_bare / pull_ms)
        print(f"Pulls that fit in 1 forward pass:    ~{affordable}")
        if t_gen:
            per_step = t_gen / 3
            print(f"Per generation step:                 {per_step:.2f} ms")
            print(f"Pulls per gen step budget:           ~{int(per_step / pull_ms)}")

    print("\nDone.")
