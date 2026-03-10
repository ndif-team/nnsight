# Performance Benchmarks

This directory contains performance benchmarks comparing NNsight interventions against raw PyTorch hooks.

## Benchmarks

### `benchmark_interventions.py`

Compares the overhead of saving layer outputs at scale:
- **NNsight**: Using `.output.save()` within trace contexts
- **PyTorch Hooks**: Using `register_forward_hook()` directly

#### What it measures:
- Wall-clock time for generation with varying numbers of interventions
- Scaling across: layers (1-12), tokens (1-20), module types (attn, mlp, both)

#### Run:
```bash
cd nnsight/tests/performance
python benchmark_interventions.py
```

#### Output:
- Console summary table with timing comparisons
- Plots saved to `results/`:
  - `interventions_scaling.png` - Time vs number of interventions
  - `overhead_ratio.png` - NNsight/PyTorch overhead ratio
  - `by_module_type.png` - Breakdown by attention vs MLP
  - `by_tokens.png` - Breakdown by token count

## Configuration

Edit the constants at the top of each benchmark file:
- `MODEL_NAME`: HuggingFace model to use
- `DEVICE`: "cuda" or "cpu"
- `WARMUP_RUNS`: Number of warmup iterations
- `BENCHMARK_RUNS`: Number of timed iterations
- `NUM_LAYERS_OPTIONS`: Layer counts to test
- `NUM_TOKENS_OPTIONS`: Token counts to test
