# vLLM intervention-gap probes

Standalone diagnostic scripts that compare how the **vLLM** backend and the
**HuggingFace** backend expose module internals to NNsight interventions. Each
`test_<group>_<n>.py` isolates one "gap" — a place where vLLM's fused kernels,
dual residual stream, merged projections, or `inference_mode` change what a
module returns or whether an operation is possible.

These are **not** pytest unit tests; they're executable probes that print a JSON
verdict per (test, backend). `run_all.py` orchestrates them, running vLLM and HF
in separate subprocesses on separate GPUs and printing a comparison table.

```bash
python tests/vllm_intervention_gaps/run_all.py --vllm-gpu 0 --hf-gpu 1
python tests/vllm_intervention_gaps/run_all.py --vllm-gpu 0 --test 1_2   # vLLM only, one gap
```

| Group | Gaps |
|-------|------|
| 1 | activation semantics: clone-on-save, dual-stream output, position-id input, fused-RMSNorm tuple, norm input arity |
| 2 | module architecture: merged `gate_up_proj` / `qkv_proj`, `RowParallelLinear` tuple |
| 3 | data layout: flat `[total_tokens, hidden]`, no PagedAttention weights |
| 4 | advanced: gradients blocked, fused-kernel source tracing, module-skip tuple |
| — | `test_stop_and_errors.py`: deferred-exception / engine-survival behavior |

The **user-facing** writeups that used to live beside these scripts
(`REPORT.md`, `VLLM_GUIDE.md`) have been folded into
[`docs/models/vllm.md`](../../docs/models/vllm.md), which reflects current
behavior. These probes predate that doc (some were authored against vLLM 0.15.1),
so treat `docs/models/vllm.md` as the source of truth and re-run a probe if you
need to confirm a specific claim on the current vLLM version.
