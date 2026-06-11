# Manual PP harnesses (not collected by pytest)

Standalone multi-GPU scripts for the vLLM pipeline-parallel path. Each is run
directly (`CUDA_VISIBLE_DEVICES=… python tests/vllm/pp/manual/<script>`), needs
2+ free GPUs, and prints its own pass/fail report — see each script's docstring
for the exact invocation. The pytest suites live one level up in
`tests/vllm/pp/`.

| Script | What it does |
|---|---|
| `pull_e2e.py` | End-to-end cross-stage pull scenarios (single/multi-token, reads + writes) |
| `profile_and_corner_cases.py` | PP=1 vs PP=2 overhead profiling + lazy-proxy/listener corner cases |
| `run_comparison.py` / `run_profile.py` | Plain-generation PP=1 vs PP=2 timing comparisons |
| `profile_pull.py` / `profile_quick.py` | Pull-protocol latency microbenchmarks |
| `measure_buffer.py` | `pp_hook_buffer` growth: leak vs allocator caching, intra-request peak |
| `stress_serve.py` / `stress_tp_serve.py` | Serve-path stress at PP=2 (and PP×TP) |
| `run_equivalence_matrix.py` | Every (TP, PP) config vs the single-GPU oracle, per model |
| `_pp_worker.py`, `_pp_pull_worker.py`, `_pp_repro_worker.py` | Subprocess workers spawned by the above and by `../test_integration.py` |
