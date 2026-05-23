"""Quick PP pull profiling — just the essentials.

Run: CUDA_VISIBLE_DEVICES=X,Y python tests/profile_pp_quick.py
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

    def profile_pull(num_pulls, shape, dtype_str="bfloat16", direction="0to1"):
        results = rpc("test_pp_profile_pull", args=(num_pulls, shape, dtype_str, direction))
        rpc("test_pp_buffer_clear")
        for r in results:
            info = pickle.loads(r)
            if info.get("role") == "puller":
                return info
        return None

    # Warmup
    profile_pull(1, [768])

    # --- Part A: Size scaling ---
    print("Part A: Pull latency vs size")
    print(f"{'Shape':<25} {'Bytes':>10} {'Mean ms':>10} {'Min ms':>10}")
    for shape in [[1], [768], [1,128,768], [1,512,768], [1,1024,768], [1,4096,768]]:
        info = profile_pull(3, shape)
        if info:
            times = info["times_ms"][1:]  # drop first
            mean = sum(times)/len(times)
            mn = min(times)
            print(f"{str(shape):<25} {info['tensor_bytes']:>10,} {mean:>10.2f} {mn:>10.2f}")

    # --- Part B: Count scaling ---
    print("\nPart B: Sequential pull scaling [1,512,768]=768KB")
    print(f"{'N':>5} {'Total ms':>10} {'Mean ms':>10}")
    for n in [1, 2, 5, 10, 20, 50]:
        info = profile_pull(n, [1,512,768])
        if info:
            print(f"{n:>5} {info['total_ms']:>10.2f} {info['mean_ms']:>10.2f}")

    # --- Part C: Trace baseline ---
    print("\nPart C: Trace overhead")
    for _ in range(3):
        with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
            model.logits.output.save()

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
            model.logits.output.save()
        times.append((time.perf_counter() - t0) * 1000)
    fwd = sum(times)/len(times)
    print(f"Forward pass (bare trace):  {fwd:.2f} ms")

    info_10 = profile_pull(10, [1,512,768])
    if info_10:
        pull = info_10["mean_ms"]
        print(f"Per pull (768KB bf16):      {pull:.2f} ms")
        print(f"Pull/forward ratio:         {pull/fwd*100:.1f}%")
        print(f"~{int(fwd/pull)} pulls fit in 1 forward pass")

    print("\nDone.")
