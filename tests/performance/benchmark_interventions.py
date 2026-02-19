"""
Performance benchmark comparing NNsight interventions vs PyTorch hooks.

This script benchmarks the overhead of saving layer outputs at scale:
- Varying number of layers
- Varying number of generation tokens
- Different NNsight configuration options

Run with: python benchmark_interventions.py
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from pathlib import Path
from itertools import product

import nnsight
from nnsight import CONFIG
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "openai-community/gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "The quick brown fox jumps over the lazy dog"
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

# Scaling parameters
NUM_LAYERS_OPTIONS = [1, 4, 8, 12]
NUM_TOKENS_OPTIONS = [1, 5, 10, 20]

# NNsight config combinations to test
# Each is a tuple: (name, {config_dict}, use_pymount_save)
# Note: PYMOUNT=True means use .output.save() syntax
#       use_pymount_save=False means use save(output) function syntax
NNSIGHT_CONFIGS = [
    (
        "baseline",
        {"PYMOUNT": True, "CROSS_INVOKER": True, "TRACE_CACHING": False},
        True,
    ),
    (
        "trace_cache",
        {"PYMOUNT": True, "CROSS_INVOKER": True, "TRACE_CACHING": True},
        True,
    ),
    (
        "no_cross",
        {"PYMOUNT": True, "CROSS_INVOKER": False, "TRACE_CACHING": False},
        True,
    ),
    (
        "no_pymount",
        {"PYMOUNT": False, "CROSS_INVOKER": True, "TRACE_CACHING": False},
        False,
    ),
    (
        "trace+no_cross",
        {"PYMOUNT": True, "CROSS_INVOKER": False, "TRACE_CACHING": True},
        True,
    ),
    (
        "trace+no_pymount",
        {"PYMOUNT": False, "CROSS_INVOKER": True, "TRACE_CACHING": True},
        False,
    ),
    (
        "minimal",
        {"PYMOUNT": False, "CROSS_INVOKER": False, "TRACE_CACHING": False},
        False,
    ),
    ("all_opts", {"PYMOUNT": True, "CROSS_INVOKER": True, "TRACE_CACHING": True}, True),
]


@dataclass
class BenchmarkResult:
    """Result from a single benchmark configuration."""

    num_interventions: int
    num_layers: int
    num_tokens: int
    config_name: str
    time_ms: float
    std_ms: float


# =============================================================================
# PyTorch Hooks Baseline
# =============================================================================


class PyTorchHookBenchmark:
    """Benchmark using raw PyTorch hooks."""

    def __init__(self, model: torch.nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.saved_outputs: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []

    def _make_hook(self, name: str):
        """Create a hook that saves output to the dictionary."""

        def hook(module, input, output):
            if name not in self.saved_outputs:
                self.saved_outputs[name] = []
            if isinstance(output, tuple):
                self.saved_outputs[name].append(output[0].detach().clone())
            else:
                self.saved_outputs[name].append(output.detach().clone())

        return hook

    def register_hooks(self, num_layers: int):
        """Register hooks on attention and MLP for specified layers."""
        self.clear_hooks()
        self.saved_outputs = {}

        for i in range(num_layers):
            layer = self.model.transformer.h[i]
            hook = layer.attn.register_forward_hook(self._make_hook(f"layer{i}_attn"))
            self.hooks.append(hook)
            hook = layer.mlp.register_forward_hook(self._make_hook(f"layer{i}_mlp"))
            self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.saved_outputs = {}

    @torch.no_grad()
    def benchmark_generation(self, num_tokens: int, num_layers: int) -> float:
        """Run generation and return time in milliseconds."""
        inputs = self.tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        self.register_hooks(num_layers)

        input_ids = inputs["input_ids"]
        for _ in range(num_tokens):
            outputs = self.model(input_ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        self.clear_hooks()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) * 1000


# =============================================================================
# NNsight Benchmark Functions
# =============================================================================

# Global model reference
_nnsight_model = None


def get_nnsight_model() -> nnsight.LanguageModel:
    """Get or create the NNsight model singleton."""
    global _nnsight_model
    if _nnsight_model is None:
        _nnsight_model = nnsight.LanguageModel(
            MODEL_NAME, device_map=DEVICE, dispatch=True
        )
    return _nnsight_model


def set_nnsight_config(config_dict: dict):
    """Set NNsight CONFIG options."""
    # Apply specified options
    for key, value in config_dict.items():
        setattr(CONFIG.APP, key, value)


def nnsight_benchmark_method_save(num_layers: int, num_tokens: int) -> float:
    """Benchmark using .output.save() method syntax (requires PYMOUNT=True)."""
    model = get_nnsight_model()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with model.generate(
        PROMPT, max_new_tokens=num_tokens, pad_token_id=50256
    ) as generator:
        with generator.iter[:] as token_idx:
            for layer_idx in range(num_layers):
                _ = model.transformer.h[layer_idx].attn.output.save()
                _ = model.transformer.h[layer_idx].mlp.output.save()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000


def nnsight_benchmark_function_save(num_layers: int, num_tokens: int) -> float:
    """Benchmark using save(output) function syntax (works with PYMOUNT=False)."""
    from nnsight import save

    model = get_nnsight_model()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with model.generate(
        PROMPT, max_new_tokens=num_tokens, pad_token_id=50256
    ) as generator:
        with generator.iter[:] as token_idx:
            for layer_idx in range(num_layers):
                _ = save(model.transformer.h[layer_idx].attn.output)
                _ = save(model.transformer.h[layer_idx].mlp.output)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000


# =============================================================================
# Benchmarking Functions
# =============================================================================


def calculate_num_interventions(num_layers: int, num_tokens: int) -> int:
    """Calculate total number of interventions (attn + mlp per layer per token)."""
    return num_layers * num_tokens * 2


def run_config_benchmark(
    config_name: str,
    config_dict: dict,
    use_pymount: bool,
    num_layers: int,
    num_tokens: int,
) -> BenchmarkResult:
    """Run benchmark for a specific NNsight config."""
    num_interventions = calculate_num_interventions(num_layers, num_tokens)

    # Set config
    set_nnsight_config(config_dict)

    # Select benchmark function based on PYMOUNT setting
    # use_pymount=True means PYMOUNT=True, so .save() method works
    # use_pymount=False means PYMOUNT=False, so we need save() function
    bench_fn = (
        nnsight_benchmark_method_save
        if use_pymount
        else nnsight_benchmark_function_save
    )

    # Warmup
    for _ in range(WARMUP_RUNS):
        bench_fn(num_layers, num_tokens)

    # Benchmark
    times = []
    for _ in range(BENCHMARK_RUNS):
        t = bench_fn(num_layers, num_tokens)
        times.append(t)

    return BenchmarkResult(
        num_interventions=num_interventions,
        num_layers=num_layers,
        num_tokens=num_tokens,
        config_name=config_name,
        time_ms=np.mean(times),
        std_ms=np.std(times),
    )


def run_pytorch_benchmark(
    pytorch_bench: PyTorchHookBenchmark,
    num_layers: int,
    num_tokens: int,
) -> BenchmarkResult:
    """Run PyTorch hooks benchmark."""
    num_interventions = calculate_num_interventions(num_layers, num_tokens)

    # Warmup
    for _ in range(WARMUP_RUNS):
        pytorch_bench.benchmark_generation(num_tokens, num_layers)

    # Benchmark
    times = []
    for _ in range(BENCHMARK_RUNS):
        t = pytorch_bench.benchmark_generation(num_tokens, num_layers)
        times.append(t)

    return BenchmarkResult(
        num_interventions=num_interventions,
        num_layers=num_layers,
        num_tokens=num_tokens,
        config_name="pytorch_hooks",
        time_ms=np.mean(times),
        std_ms=np.std(times),
    )


def run_all_benchmarks() -> List[BenchmarkResult]:
    """Run all benchmark configurations."""
    print(f"Loading models on {DEVICE}...")

    # Load PyTorch model
    pytorch_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    pytorch_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pytorch_bench = PyTorchHookBenchmark(pytorch_model, tokenizer)

    # Warm up NNsight model
    _ = get_nnsight_model()

    results = []
    total_configs = len(NNSIGHT_CONFIGS) + 1  # +1 for pytorch

    print(f"\nRunning benchmarks for {total_configs} configurations...")
    print(f"Layers: {NUM_LAYERS_OPTIONS}, Tokens: {NUM_TOKENS_OPTIONS}")

    # Run PyTorch baseline
    print(f"\n[1/{total_configs}] PyTorch Hooks")
    for num_tokens in NUM_TOKENS_OPTIONS:
        for num_layers in NUM_LAYERS_OPTIONS:
            num_int = calculate_num_interventions(num_layers, num_tokens)
            print(
                f"  {num_layers} layers, {num_tokens} tokens -> {num_int} interventions"
            )
            result = run_pytorch_benchmark(pytorch_bench, num_layers, num_tokens)
            results.append(result)

    # Run NNsight configs
    for idx, (config_name, config_dict, use_pymount) in enumerate(NNSIGHT_CONFIGS):
        print(f"\n[{idx+2}/{total_configs}] NNsight: {config_name}")
        for num_tokens in NUM_TOKENS_OPTIONS:
            for num_layers in NUM_LAYERS_OPTIONS:
                num_int = calculate_num_interventions(num_layers, num_tokens)
                print(
                    f"  {num_layers} layers, {num_tokens} tokens -> {num_int} interventions"
                )
                result = run_config_benchmark(
                    config_name, config_dict, use_pymount, num_layers, num_tokens
                )
                results.append(result)

    return results


# =============================================================================
# Plotting
# =============================================================================


def plot_results(results: List[BenchmarkResult], output_dir: Path):
    """Generate plots from benchmark results."""
    output_dir.mkdir(exist_ok=True)

    # Get unique config names maintaining order
    config_names = ["pytorch_hooks"] + [c[0] for c in NNSIGHT_CONFIGS]

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    color_map = dict(zip(config_names, colors))

    # Plot 1: Time vs Interventions for all configs
    fig, ax = plt.subplots(figsize=(12, 7))

    for config_name in config_names:
        config_results = [r for r in results if r.config_name == config_name]
        config_results.sort(key=lambda r: r.num_interventions)

        interventions = [r.num_interventions for r in config_results]
        times = [r.time_ms for r in config_results]

        ax.plot(
            interventions,
            times,
            "o-",
            label=config_name,
            color=color_map[config_name],
            markersize=6,
            linewidth=2,
        )

    ax.set_xlabel("Number of Interventions", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(
        "NNsight Configs vs PyTorch Hooks: Performance Comparison", fontsize=14
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "config_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'config_comparison.png'}")

    # Plot 2: Overhead ratio for each config vs PyTorch
    fig, ax = plt.subplots(figsize=(12, 7))

    pytorch_results = {
        (r.num_layers, r.num_tokens): r.time_ms
        for r in results
        if r.config_name == "pytorch_hooks"
    }

    for config_name in config_names[1:]:  # Skip pytorch_hooks
        config_results = [r for r in results if r.config_name == config_name]
        config_results.sort(key=lambda r: r.num_interventions)

        interventions = [r.num_interventions for r in config_results]
        overheads = [
            r.time_ms / pytorch_results[(r.num_layers, r.num_tokens)]
            for r in config_results
        ]

        ax.plot(
            interventions,
            overheads,
            "o-",
            label=config_name,
            color=color_map[config_name],
            markersize=6,
            linewidth=2,
        )

    ax.axhline(
        y=1.0, color="red", linestyle="--", linewidth=2, label="No overhead (1.0x)"
    )
    ax.set_xlabel("Number of Interventions", fontsize=12)
    ax.set_ylabel("Overhead Ratio (Config / PyTorch)", fontsize=12)
    ax.set_title("NNsight Config Overhead Relative to PyTorch Hooks", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "overhead_by_config.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'overhead_by_config.png'}")

    # Plot 3: Bar chart of average overhead by config
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_overheads = []
    for config_name in config_names[1:]:
        config_results = [r for r in results if r.config_name == config_name]
        overheads = [
            r.time_ms / pytorch_results[(r.num_layers, r.num_tokens)]
            for r in config_results
        ]
        avg_overheads.append(np.mean(overheads))

    bars = ax.bar(
        config_names[1:], avg_overheads, color=[color_map[c] for c in config_names[1:]]
    )
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="No overhead")
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Average Overhead (x PyTorch)", fontsize=12)
    ax.set_title("Average Overhead by NNsight Configuration", fontsize=14)
    ax.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, val in zip(bars, avg_overheads):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "avg_overhead_bar.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'avg_overhead_bar.png'}")


def print_summary(results: List[BenchmarkResult]):
    """Print a summary table of results."""
    # Get pytorch baseline for overhead calculation
    pytorch_results = {
        (r.num_layers, r.num_tokens): r.time_ms
        for r in results
        if r.config_name == "pytorch_hooks"
    }

    print("\n" + "=" * 95)
    print("BENCHMARK SUMMARY")
    print("=" * 95)
    print(
        f"{'Config':<15} | {'Layers':>6} | {'Tokens':>6} | {'Intv':>6} | "
        f"{'Time (ms)':>10} | {'vs PyTorch':>10}"
    )
    print("-" * 95)

    config_names = ["pytorch_hooks"] + [c[0] for c in NNSIGHT_CONFIGS]

    for config_name in config_names:
        config_results = [r for r in results if r.config_name == config_name]
        config_results.sort(key=lambda r: (r.num_tokens, r.num_layers))

        for r in config_results:
            pytorch_time = pytorch_results.get((r.num_layers, r.num_tokens), r.time_ms)
            overhead = r.time_ms / pytorch_time if pytorch_time > 0 else 0
            overhead_str = f"{overhead:.2f}x" if config_name != "pytorch_hooks" else "-"
            print(
                f"{config_name:<15} | {r.num_layers:>6} | {r.num_tokens:>6} | "
                f"{r.num_interventions:>6} | {r.time_ms:>10.2f} | {overhead_str:>10}"
            )

    print("=" * 95)

    # Average overheads
    print("\nAverage Overheads:")
    for config_name in config_names[1:]:
        config_results = [r for r in results if r.config_name == config_name]
        overheads = [
            r.time_ms / pytorch_results[(r.num_layers, r.num_tokens)]
            for r in config_results
        ]
        print(f"  {config_name:<15}: {np.mean(overheads):.2f}x")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("NNsight Config Performance Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")
    print(f"Layer options: {NUM_LAYERS_OPTIONS}")
    print(f"Token options: {NUM_TOKENS_OPTIONS}")
    print(f"\nConfigs to test:")
    for name, config, pymount in NNSIGHT_CONFIGS:
        print(f"  - {name}: {config}, pymount={pymount}")
    print()

    # Run benchmarks
    results = run_all_benchmarks()

    # Print summary
    print_summary(results)

    # Generate plots
    output_dir = Path(__file__).parent / "results"
    plot_results(results, output_dir)

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
