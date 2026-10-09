## Findings

**Headline: on the production cache (Lustre /work/nvme, cold), run:ai GPU-direct loads 2.3x to 4.5x faster than HF `from_pretrained` on every GPU type, for every model.** Best cold Lustre times, GPU-direct (or lazy) vs HF mmap:

| Model | A40 | A100 | H200 |
|---|---|---|---|
| Qwen2.5-7B-Instruct (14 GB) | 5.2 s vs 16.8 s (3.2x) | 5.4 s vs 16.9 s (3.1x) | 2.8 s vs 8.4 s (3.0x) |
| Qwen3-32B (61 GB) | 25.2 s vs 63.3 s (2.5x) | 27.1 s vs 61.9 s (2.3x) | 14.5 s vs 36.6 s (2.5x) |
| Qwen3-30B-A3B (57 GB MoE) | 28.5 s vs 105.1 s (3.7x) | 30.1 s vs 103.6 s (3.4x) | 17.3 s vs 77.7 s (4.5x) |

1. **HF has not closed the gap.** transformers 5.17 loads tensors exactly as 5.3 did (mmap slice, blocking per-tensor `.to(device)`, 4 worker threads). Its peak effective bandwidth from cold Lustre is 0.5 to 1.7 GB/s; GPU-direct reaches 2.3 to 3.5 GB/s on the same storage.
2. **The safetensors `pread` backend is not a substitute.** Forced on (transformers only uses it on MPS/Windows) it helps the MoE model on cold Lustre (1.7x to 2.1x) but is equal or *slower* everywhere else, and catastrophically slow (0.1x to 0.3x) on a warm local cache because it re-reads what mmap would serve from RAM. Not worth enabling by default.
3. **The MoE model is HF's worst case.** Qwen3-30B-A3B takes HF 104 s from cold Lustre on A40/A100, versus 62 s for the 7% larger dense Qwen3-32B: the expert-merge conversion (`MergeModulelist`) serialises behind the slow mmap reads. GPU-direct lazy is 3.4x to 4.5x faster there.
4. **H200 nodes are faster for every path** (roughly 1.7x for HF, 1.5 to 2x for GPU-direct), which looks like their CPU/PCIe generation rather than the GPU itself, since loading is I/O and copy bound.
5. **Where HF catches up: warm node-local /tmp.** With the whole model in the page cache on the node, HF mmap reaches 4.5 to 6.3 GB/s and GPU-direct is only 0.9x to 1.2x faster. That state is rare in production (fresh node, first load) and reaching it costs a 2 to 3 minute copy per 60 GB model.
6. **Knobs.** concurrency 4 is the best or within noise of best for run:ai in most cells; 8 wins on node-local /tmp. `lazy` is the best variant for the MoE model on cold Lustre and otherwise ties GPU-direct.
7. **Memory.** HF stays under 0.5 GB private CPU (mmap pages are shared); GPU-direct peaks at 6 to 9 GB, lazy at 9 to 11 GB, CPU-clone stream at 12 to 20 GB. All fit comfortably in a 200 GB job.
8. **Correctness.** Every path produced bit-identical logits to HF `from_pretrained` on both storage tiers for all three models (0 of 36 cold-pass verifications mismatched).
9. **Noise.** Some warm-Lustre cells are erratic (e.g. A40 Qwen3-30B-A3B HF warm 96 s vs A100 43 s; lazy 42 s vs GPU-direct 14 s) because the Lustre page cache is shared with other jobs and can be evicted between warmup and the timed load. Cold-cache rows are the reliable baseline; single repeat per config.

## Change vs the March 2026 measurements

On A40 with cold Lustre, HF got slower on the same models (Qwen3-30B-A3B: 74 s then, 104 s now; Qwen3-32B: 53 s then, 63 s now) while GPU-direct held or improved (Qwen3-30B-A3B: 27 s then, 28 s now; Qwen3-32B: 26 s then, 25 s now). The speedup widened from ~2x to 2.5x to 3.7x. The Lustre filesystem is at 92% capacity and shared, so the HF drift may be storage-side rather than transformers-side.

## Caveats

- One repeat per config; the sweep already costs ~1 h per (GPU, model). Re-run with `--repeats 3` on cold cache if a cell needs a tighter number.
- Qwen2.5-7B-Instruct stands in for the uncached Qwen3-8B.
- The A100 nodes on Delta are 40 GB SXM4; 2 GPUs still hold every model here.
- The whole run charged ~14 GPU-hours to bdnh-delta-gpu on the plain partitions.
