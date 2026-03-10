"""
NNsight Performance Profiling Framework

This module provides comprehensive profiling tools for analyzing
NNsight's performance characteristics.

Usage:
    # Run full profiling
    python run_full_profile.py

    # Run individual profilers
    python profile_tracing.py
    python profile_interleaving.py
    python profile_envoy.py
    python profile_batching.py
"""

from .profiler_utils import (
    TimingResult,
    MemoryResult,
    ProfileResult,
    InstrumentedTimer,
    time_function,
    profile_function,
    force_gc,
    sync_cuda,
)
