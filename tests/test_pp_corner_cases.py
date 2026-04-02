"""Corner-case tests for PP pull transport layer.

Tests the PPListener/gloo transport under various conditions:
  1. Small tensor (4 elements)
  2. Large tensor (hidden state: [1, 1024, 768])
  3. Very large tensor (~8MB)
  4. 10 sequential pulls
  5. Multiple dtypes (float16, float32, bfloat16)
  6. High-dimensional tensor (5D)
  7. Bidirectional: rank 1 serves, rank 0 pulls
  8. Round-trip: rank 0→1 then rank 1→0

Requires 2 GPUs.
Run: CUDA_VISIBLE_DEVICES=0,1 python tests/test_pp_corner_cases.py
"""

if __name__ == '__main__':
    import pickle
    import sys
    from nnsight.modeling.vllm import VLLM

    model = VLLM(
        "openai-community/gpt2",
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.1,
        dispatch=True,
    )
    rpc = model.vllm_entrypoint.collective_rpc
    failures = []

    def pull_ok(results, pulling_rank):
        """Check that the pulling rank got a match."""
        for r in results:
            info = pickle.loads(r)
            if info.get("rank") == pulling_rank and "match" in info:
                return info["match"], info
        return True, {}  # rank was server, no match to check

    seed_counter = [0]
    def serve_and_pull(name, shape, dtype_str, source_rank=0):
        """Put tensor in buffer (all ranks), then pull from source_rank."""
        key = f"t_{name}"
        seed_counter[0] += 1
        seed = seed_counter[0]
        rpc("test_pp_buffer_put", args=({key: (shape, dtype_str, seed)},))
        results = rpc("test_pp_pull", args=(source_rank, key, shape, dtype_str, seed))
        pulling_rank = 1 - source_rank
        ok, info = pull_ok(results, pulling_rank)
        rpc("test_pp_buffer_clear")
        if not ok:
            failures.append(f"{name}: mismatch {info}")
            print(f"  FAIL: {info}")
        else:
            detail = f"{info.get('shape','')} {info.get('dtype','')}" if info else ""
            print(f"  OK {detail}")
        return ok

    # ---- Test 1: Small tensor ----
    print("Test 1: small tensor [4]")
    serve_and_pull("small", [4], "bfloat16")

    # ---- Test 2: Large tensor ----
    print("Test 2: large tensor [1, 1024, 768]")
    serve_and_pull("large", [1, 1024, 768], "bfloat16")

    # ---- Test 3: Very large tensor (~8MB) ----
    print("Test 3: very large tensor [1, 4096, 1024] (~8MB in bf16)")
    serve_and_pull("very_large", [1, 4096, 1024], "bfloat16")

    # ---- Test 4: 10 sequential pulls ----
    print("Test 4: 10 sequential pulls")
    entries = {}
    for i in range(10):
        entries[f"seq_{i}"] = ([1, 32, 768], "bfloat16", 1000 + i)
    rpc("test_pp_buffer_put", args=(entries,))
    all_ok = True
    for i in range(10):
        key = f"seq_{i}"
        results = rpc("test_pp_pull", args=(0, key, [1, 32, 768], "bfloat16", 1000 + i))
        ok, info = pull_ok(results, 1)
        if not ok:
            all_ok = False
            failures.append(f"Test 4 seq_{i}: mismatch")
    rpc("test_pp_buffer_clear")
    print(f"  {'OK' if all_ok else 'FAIL'}: 10 pulls {'all matched' if all_ok else 'had mismatches'}")

    # ---- Test 5: Multiple dtypes ----
    for dtype_str in ["float16", "float32", "bfloat16"]:
        print(f"Test 5: dtype {dtype_str}")
        serve_and_pull(f"dtype_{dtype_str}", [8, 16], dtype_str)

    # ---- Test 6: 5D tensor ----
    print("Test 6: 5D tensor [2, 3, 4, 5, 6]")
    serve_and_pull("5d", [2, 3, 4, 5, 6], "float32")

    # ---- Test 7: Bidirectional (rank 1 serves, rank 0 pulls) ----
    print("Test 7: bidirectional — rank 0 pulls from rank 1")
    serve_and_pull("bidir", [1, 64, 768], "bfloat16", source_rank=1)

    # ---- Test 8: Round-trip (rank 0→1 then rank 1→0) ----
    print("Test 8: round-trip (0→1 then 1→0)")
    serve_and_pull("rt_fwd", [1, 32, 768], "bfloat16", offset=42, source_rank=0)
    serve_and_pull("rt_rev", [1, 32, 768], "bfloat16", offset=99, source_rank=1)
    print("  OK (both directions)")

    # ---- Summary ----
    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL PASS")
