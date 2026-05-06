# nnsight-serve Client Test Scripts

10 standalone client scripts for testing `nnsight-serve` manually. Each simulates a different user workload — run them individually, in parallel, or mix and match to stress-test the server.

**Model:** `Qwen/Qwen3-30B-A3B` (48 layers, 2048 hidden dim, 128 experts (8 active per token))

## Start the server

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m nnsight.modeling.vllm.serve.cli Qwen/Qwen3-30B-A3B \
    --port 6677 \
    --gpu-memory-utilization 0.5 \
    --tensor-parallel-size 2
```

Or single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m nnsight.modeling.vllm.serve.cli Qwen/Qwen3-30B-A3B \
    --port 6677 \
    --gpu-memory-utilization 0.8
```

Wait for `Uvicorn running on http://127.0.0.1:6677` before running clients.

## Run a single client

```bash
python tests/serve_clients/client_01_basic_logits.py
```

## Run all 10 concurrently

```bash
for f in tests/serve_clients/client_*.py; do python "$f" & done; wait
```

## Run N copies of the same client

```bash
for i in $(seq 1 5); do python tests/serve_clients/client_03_intervention.py & done; wait
```

## Client summary

| # | File | What it does |
|---|------|-------------|
| 01 | `client_01_basic_logits.py` | Capture logits, decode next token |
| 02 | `client_02_multi_layer.py` | Save activations from 6 layers |
| 03 | `client_03_intervention.py` | Zero out MLP, verify prediction changes |
| 04 | `client_04_multi_invoke.py` | 3 prompts in one trace via invoke() |
| 05 | `client_05_shared_list.py` | Shared list across invokes |
| 06 | `client_06_nonblocking.py` | 4 concurrent non-blocking requests |
| 07 | `client_07_activation_patch.py` | Patch activations from one prompt into another |
| 08 | `client_08_attention.py` | Capture attention layer outputs |
| 09 | `client_09_expert_outputs.py` | Capture MoE expert/gate outputs |
| 10 | `client_10_stress.py` | Rapid-fire 20 blocking requests |

## Environment variables

- `NNSIGHT_SERVE_URL` — server URL (default: `http://127.0.0.1:6677`)
- `CUDA_VISIBLE_DEVICES` — GPU for meta model init (clients use no GPU memory)
