"""
Profiling utilities for NNsight performance analysis.

This module provides common utilities and decorators for profiling
different aspects of NNsight's execution.
"""

import cProfile
import gc
import io
import pstats
import time
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import tracemalloc

import torch


@dataclass
class TimingResult:
    """Container for timing results."""

    name: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float = 0.0
    individual_times_ms: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.name}: mean={self.mean_time_ms:.3f}ms, "
            f"min={self.min_time_ms:.3f}ms, max={self.max_time_ms:.3f}ms, "
            f"total={self.total_time_ms:.3f}ms ({self.iterations} iterations)"
        )


@dataclass
class MemoryResult:
    """Container for memory profiling results."""

    name: str
    peak_memory_mb: float
    current_memory_mb: float
    traced_memory_mb: float

    def __str__(self) -> str:
        return (
            f"{self.name}: peak={self.peak_memory_mb:.2f}MB, "
            f"current={self.current_memory_mb:.2f}MB"
        )


@dataclass
class ProfileResult:
    """Container for cProfile results."""

    name: str
    stats_str: str
    top_functions: List[
        Tuple[str, float, int]
    ]  # (function_name, cumulative_time, calls)
    total_time: float

    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Profile: {self.name}",
            f"Total time: {self.total_time:.4f}s",
            "",
        ]
        lines.append("Top functions by cumulative time:")
        for func, time_s, calls in self.top_functions[:10]:
            lines.append(f"  {time_s:.4f}s ({calls} calls): {func}")
        return "\n".join(lines)


def force_gc():
    """Force garbage collection with multiple cycles."""
    for _ in range(3):
        gc.collect()


def sync_cuda():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    sync_cuda()
    start = time.perf_counter()
    yield
    sync_cuda()
    elapsed = (time.perf_counter() - start) * 1000  # ms
    return elapsed


def time_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    iterations: int = 10,
    warmup: int = 2,
    name: str = None,
    verbose: bool = False,
) -> TimingResult:
    """
    Time a function over multiple iterations.

    Args:
        func: Function to time
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)
        name: Name for the timing result

    Returns:
        TimingResult with timing statistics
    """
    if kwargs is None:
        kwargs = {}
    if name is None:
        name = getattr(func, "__name__", str(func))

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
        force_gc()

    # Timed iterations
    times = []
    for _ in range(iterations):
        sync_cuda()
        start = time.perf_counter()
        func(*args, **kwargs)
        sync_cuda()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
        force_gc()

    total = sum(times)
    mean = total / len(times)
    min_t = min(times)
    max_t = max(times)

    # Calculate std dev
    if len(times) > 1:
        variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
        std_dev = variance**0.5
    else:
        std_dev = 0.0

    return TimingResult(
        name=name,
        iterations=iterations,
        total_time_ms=total,
        mean_time_ms=mean,
        min_time_ms=min_t,
        max_time_ms=max_t,
        std_dev_ms=std_dev,
        individual_times_ms=times,
    )


@contextmanager
def memory_tracker():
    """Context manager for tracking memory allocation."""
    force_gc()
    tracemalloc.start()
    yield
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return MemoryResult(
        name="",
        peak_memory_mb=peak / (1024 * 1024),
        current_memory_mb=current / (1024 * 1024),
        traced_memory_mb=current / (1024 * 1024),
    )


def profile_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    name: str = None,
    sort_by: str = "cumulative",
) -> ProfileResult:
    """
    Profile a function using cProfile.

    Args:
        func: Function to profile
        args: Positional arguments
        kwargs: Keyword arguments
        name: Name for the profile result
        sort_by: Sort key for stats

    Returns:
        ProfileResult with profiling information
    """
    if kwargs is None:
        kwargs = {}
    if name is None:
        name = getattr(func, "__name__", str(func))

    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()

    # Capture stats as string
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(sort_by)
    stats.print_stats(50)
    stats_str = stream.getvalue()

    # Extract top functions
    stats.sort_stats(sort_by)
    top_functions = []
    for key, value in sorted(
        stats.stats.items(),
        key=lambda x: x[1][3],  # Sort by cumulative time
        reverse=True,
    )[:20]:
        filename, lineno, func_name = key
        cumtime = value[3]
        calls = value[0]
        top_functions.append((f"{func_name} ({filename}:{lineno})", cumtime, calls))

    total_time = stats.total_tt

    return ProfileResult(
        name=name,
        stats_str=stats_str,
        top_functions=top_functions,
        total_time=total_time,
    )


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


class InstrumentedTimer:
    """
    Timer class that can be used to instrument specific code sections.
    """

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str):
        """Start timing a section."""
        sync_cuda()
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        """Stop timing a section and record the elapsed time."""
        sync_cuda()
        elapsed = (time.perf_counter() - self._start_times[name]) * 1000
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

    @contextmanager
    def section(self, name: str):
        """Context manager for timing a section."""
        self.start(name)
        yield
        self.stop(name)

    def get_results(self) -> Dict[str, TimingResult]:
        """Get timing results for all sections."""
        results = {}
        for name, times in self.timings.items():
            total = sum(times)
            mean = total / len(times) if times else 0
            min_t = min(times) if times else 0
            max_t = max(times) if times else 0

            if len(times) > 1:
                variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
                std_dev = variance**0.5
            else:
                std_dev = 0.0

            results[name] = TimingResult(
                name=name,
                iterations=len(times),
                total_time_ms=total,
                mean_time_ms=mean,
                min_time_ms=min_t,
                max_time_ms=max_t,
                std_dev_ms=std_dev,
                individual_times_ms=times,
            )
        return results

    def clear(self):
        """Clear all timing data."""
        self.timings.clear()
        self._start_times.clear()

    def summary(self) -> str:
        """Get a summary of all timings."""
        results = self.get_results()
        lines = ["\nTiming Summary:", "-" * 60]

        # Sort by mean time
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].mean_time_ms,
            reverse=True,
        )

        for name, result in sorted_results:
            lines.append(
                f"{name:40s}: {result.mean_time_ms:8.3f}ms "
                f"(±{result.std_dev_ms:.3f}, n={result.iterations})"
            )

        return "\n".join(lines)
