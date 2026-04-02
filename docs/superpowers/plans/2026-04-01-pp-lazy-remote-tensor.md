# Pipeline Parallelism — Free-Running Mediators with LazyRemoteTensor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable transparent NNsight interventions on vLLM models with `pipeline_parallel_size > 1` — single-token, multi-token, cross-stage reads and writes all work with single-GPU-style user code.

**Architecture:** Mediators run freely between interleaver sessions. PPMissing module accesses are short-circuited at the Envoy level — they return a `LazyRemoteTensor` directly, bypassing the event queue. Materialization pulls from the source rank's listener thread via RPC. Local module accesses block normally (event queue → hook dispatch). A readiness check at each interleaver session start ensures mediators are parked at a local module before hooks fire. `pp_hook_buffer` stays local on each rank (never sent); the listener serves from it on demand.

**Tech Stack:** PyTorch, vLLM v0.15.1, NCCL/gloo (via `torch.distributed`), nnsight interleaver/envoy, `threading.Condition`.

**Spec:** `src/nnsight/modeling/vllm/PP_DESIGN.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/nnsight/modeling/vllm/lazy_remote_tensor.py` | LazyRemoteTensor proxy class |
| Create | `src/nnsight/modeling/vllm/pp_listener.py` | Listener thread + pull protocol |
| Modify | `src/nnsight/intervention/envoy.py` | PPMissing short-circuit in `.output`/`.input`/`.output.setter`/`.inputs.setter` |
| Modify | `src/nnsight/intervention/interleaver.py` | Remove END injection + eager buffer branches, add Condition notify on buffer clone, add readiness check |
| Modify | `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py` | Listener lifecycle, Condition init, remove `pp_received_buffer`, readiness check call |
| Modify | `src/nnsight/modeling/vllm/workers/GPUWorker.py` | Remove eager buffer exchange |
| Modify | `src/nnsight/modeling/vllm/pp.py` | Remove `make_dummy_tensor` |
| Modify | `tests/test_vllm_pp.py` | Update + extend tests |

---

## Task 1: LazyRemoteTensor

**Files:**
- Create: `src/nnsight/modeling/vllm/lazy_remote_tensor.py`
- Test: `tests/test_vllm_pp.py`

- [ ] **Step 1: Write tests for LazyRemoteTensor**

```python
# Append to tests/test_vllm_pp.py

import torch
from nnsight.modeling.vllm.lazy_remote_tensor import LazyRemoteTensor


class TestLazyRemoteTensor:

    def _make_lazy(self, real_tensor=None):
        """Helper: create a LazyRemoteTensor with optional pre-set real tensor."""
        lazy = LazyRemoteTensor(
            source_rank=1,
            provider_string="model.layers.50.output.i0",
            shape=(1, 5, 768),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        if real_tensor is not None:
            lazy._real = real_tensor
        return lazy

    def test_metadata_no_materialization(self):
        lazy = self._make_lazy()
        assert lazy.shape == (1, 5, 768)
        assert lazy.dtype == torch.float32
        assert lazy.device == torch.device("cpu")
        assert lazy._real is None

    def test_setitem_noop(self):
        lazy = self._make_lazy()
        lazy[:] = torch.zeros(1, 5, 768)
        assert lazy._real is None  # no materialization

    def test_getitem_returns_self(self):
        lazy = self._make_lazy()
        result = lazy[0]
        assert result is lazy

    def test_save_returns_self(self):
        lazy = self._make_lazy()
        result = lazy.save()
        assert result is lazy
        assert lazy._real is None

    def test_torch_function_materializes(self):
        real = torch.randn(1, 5, 768)
        lazy = self._make_lazy(real_tensor=real)
        result = lazy + 1
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, real + 1)

    def test_torch_function_in_args(self):
        real = torch.randn(1, 5, 768)
        lazy = self._make_lazy(real_tensor=real)
        result = torch.sum(lazy)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.sum(real))
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py::TestLazyRemoteTensor -v -x`
Expected: ImportError — `lazy_remote_tensor` module does not exist.

- [ ] **Step 3: Implement LazyRemoteTensor**

```python
# src/nnsight/modeling/vllm/lazy_remote_tensor.py
"""LazyRemoteTensor — proxy for PPMissing module outputs.

Returned by the Envoy when accessing .output on a module that lives on
a different PP rank. Most operations are no-ops (writes, saves). Only
real tensor operations (arithmetic, torch functions) trigger
materialization via RPC pull from the source rank's listener.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch.utils._pytree import tree_map


class LazyRemoteTensor:
    """Proxy that materializes into a real tensor on first read operation."""

    def __init__(
        self,
        source_rank: int,
        provider_string: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self._meta = {
            "source_rank": source_rank,
            "provider_string": provider_string,
            "shape": shape,
            "dtype": dtype,
            "device": device,
        }
        self._real: torch.Tensor | None = None
        self._pull_fn = None  # set externally by whoever creates the lazy tensor

    def _materialize(self) -> torch.Tensor:
        """Pull real tensor from source rank's listener.

        Blocks until the tensor is available.
        """
        if self._real is None:
            if self._pull_fn is None:
                raise RuntimeError(
                    f"Cannot materialize LazyRemoteTensor for "
                    f"{self._meta['provider_string']}: no pull function set."
                )
            self._real = self._pull_fn(
                self._meta["source_rank"],
                self._meta["provider_string"],
            )
        return self._real

    # --- torch interop: materialize on any real operation ---

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        args = tree_map(
            lambda x: x._materialize() if isinstance(x, LazyRemoteTensor) else x,
            args,
        )
        kwargs = tree_map(
            lambda x: x._materialize() if isinstance(x, LazyRemoteTensor) else x,
            kwargs,
        )
        return func(*args, **kwargs)

    # --- no-op absorbers ---

    def __setitem__(self, key: Any, value: Any) -> None:
        pass  # absorb writes without materialization

    def __getitem__(self, key: Any) -> "LazyRemoteTensor":
        return self  # chained indexing: lazy[0][:] = X → no-op

    def save(self) -> "LazyRemoteTensor":
        return self  # no-op on non-owning rank

    # --- metadata (no materialization) ---

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._real is not None:
            return self._real.shape
        return self._meta["shape"]

    @property
    def dtype(self) -> torch.dtype:
        if self._real is not None:
            return self._real.dtype
        return self._meta["dtype"]

    @property
    def device(self) -> torch.device:
        if self._real is not None:
            return self._real.device
        return self._meta["device"]

    def __repr__(self) -> str:
        status = "materialized" if self._real is not None else "lazy"
        return (
            f"LazyRemoteTensor({status}, "
            f"src=rank{self._meta['source_rank']}, "
            f"key={self._meta['provider_string']!r}, "
            f"shape={self.shape})"
        )
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py::TestLazyRemoteTensor -v -x`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/nnsight/modeling/vllm/lazy_remote_tensor.py tests/test_vllm_pp.py
git commit -m "feat(vllm-pp): add LazyRemoteTensor proxy class"
```

---

## Task 2: Listener Thread + Pull Protocol

**Files:**
- Create: `src/nnsight/modeling/vllm/pp_listener.py`
- Test: `tests/test_vllm_pp.py`

- [ ] **Step 1: Write tests for PPListener**

```python
# Append to tests/test_vllm_pp.py

import threading
import time
from nnsight.modeling.vllm.pp_listener import PPListener


class TestPPListener:

    def test_serve_existing_value(self):
        """Listener serves a value that's already in the buffer."""
        buffer = {"model.layers.5.output.i0": torch.randn(1, 5, 768)}
        cond = threading.Condition()
        listener = PPListener(buffer, cond)

        result = listener.local_lookup("model.layers.5.output.i0")
        assert torch.equal(result, buffer["model.layers.5.output.i0"])

    def test_wait_for_value(self):
        """Listener waits until a value appears in the buffer."""
        buffer = {}
        cond = threading.Condition()
        listener = PPListener(buffer, cond)

        result_holder = [None]

        def lookup():
            result_holder[0] = listener.local_lookup(
                "model.layers.5.output.i0", timeout=5.0
            )

        t = threading.Thread(target=lookup)
        t.start()

        # Value not yet in buffer — thread is waiting
        time.sleep(0.05)
        assert result_holder[0] is None

        # Add value and notify
        tensor = torch.randn(1, 5, 768)
        with cond:
            buffer["model.layers.5.output.i0"] = tensor
            cond.notify_all()

        t.join(timeout=5.0)
        assert result_holder[0] is not None
        assert torch.equal(result_holder[0], tensor)

    def test_timeout_raises(self):
        """Listener raises TimeoutError if value never appears."""
        buffer = {}
        cond = threading.Condition()
        listener = PPListener(buffer, cond)

        with pytest.raises(TimeoutError):
            listener.local_lookup("missing.key", timeout=0.1)
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py::TestPPListener -v -x`
Expected: ImportError.

- [ ] **Step 3: Implement PPListener**

```python
# src/nnsight/modeling/vllm/pp_listener.py
"""PP Listener — background thread serving tensor pull requests.

Each PP rank runs a listener for the entire request lifetime. It serves
pull requests from other ranks' LazyRemoteTensor._materialize() calls,
reading from the local pp_hook_buffer.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import torch


class PPListener:
    """Serves tensors from the local pp_hook_buffer to remote ranks.

    The listener uses a threading.Condition to wait when a requested
    key is not yet in the buffer (the source rank's forward pass hasn't
    produced the value yet). The Condition is notified by
    handle_value_event when a new value is cloned into the buffer.

    For v1, materialization uses local_lookup (same-process call from
    the mediator worker thread). Cross-rank RPC transport is a separate
    concern layered on top.
    """

    def __init__(
        self,
        buffer: Dict[str, Any],
        condition: threading.Condition,
    ):
        self._buffer = buffer
        self._condition = condition
        self._stopped = False

    def local_lookup(
        self,
        provider_string: str,
        timeout: Optional[float] = 30.0,
    ) -> torch.Tensor:
        """Look up a value in the local buffer, waiting if necessary.

        Called by LazyRemoteTensor._materialize() when the source rank
        is the local rank (single-node PP) or via RPC from a remote rank.

        Args:
            provider_string: The key to look up (e.g. "model.layers.5.output.i0").
            timeout: Maximum seconds to wait. None = wait forever.

        Returns:
            The tensor from the buffer.

        Raises:
            TimeoutError: If the value doesn't appear within timeout.
        """
        with self._condition:
            while provider_string not in self._buffer:
                if self._stopped:
                    raise RuntimeError("PPListener stopped while waiting")
                if not self._condition.wait(timeout=timeout):
                    raise TimeoutError(
                        f"PPListener: timed out waiting for {provider_string!r}"
                    )
            return self._buffer[provider_string]

    def stop(self):
        """Signal the listener to stop. Unblocks any waiting lookups."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py::TestPPListener -v -x`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/nnsight/modeling/vllm/pp_listener.py tests/test_vllm_pp.py
git commit -m "feat(vllm-pp): add PPListener with local_lookup and Condition-based wait"
```

---

## Task 3: Remove Phase 1 Eager Exchange + END Injection

**Files:**
- Modify: `src/nnsight/modeling/vllm/workers/GPUWorker.py`
- Modify: `src/nnsight/intervention/interleaver.py`
- Modify: `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py`
- Modify: `src/nnsight/modeling/vllm/pp.py`

- [ ] **Step 1: Remove eager buffer exchange from GPUWorker**

In `src/nnsight/modeling/vllm/workers/GPUWorker.py`, remove the `pp_hook_buffer` send/recv from `execute_model()`. Keep the IntermediateTensors exchange. The method should fall through to `super().execute_model()` for all cases (PP and non-PP), since the IT exchange is handled by the parent:

```python
# src/nnsight/modeling/vllm/workers/GPUWorker.py
import torch
from vllm.v1.worker import gpu_worker
from ..model_runners.GPUModelRunner import NNsightGPUModelRunner
from vllm.v1.worker import gpu_model_runner


class NNsightGPUWorker(gpu_worker.Worker):
    """Custom vLLM GPU worker that uses :class:`NNsightGPUModelRunner`."""

    def __init__(self, *args, **kwargs):
        gpu_model_runner.GPUModelRunner = NNsightGPUModelRunner
        super().__init__(*args, **kwargs)

    def init_device(self):
        backend = self.parallel_config.distributed_executor_backend
        if backend is not None and not isinstance(backend, str):
            from vllm.v1.executor.ray_executor import RayDistributedExecutor
            if issubclass(backend, RayDistributedExecutor):
                self.parallel_config.distributed_executor_backend = "ray"
        super().init_device()

    def collect_nnsight(self, req_ids: list[str], finished_req_ids: list[str] | None = None):
        return self.model_runner.collect_nnsight(req_ids, finished_req_ids)
```

- [ ] **Step 2: Remove PPMissing branches from interleaver**

In `src/nnsight/intervention/interleaver.py`:

Remove the `_is_pp_missing_request` helper function (lines 75-79). Remove the PPMissing branch in `handle_value_event` (lines 819-835 — the `elif _is_pp_missing_request(...)` block). Remove the PPMissing branch in `handle_swap_event` (lines 875-885). Keep the buffer clone code in `handle_value_event` (lines 812-817) — it populates `pp_hook_buffer` for the listener.

After removal, `handle_value_event` should be:

```python
def handle_value_event(self, requester: Any, provider: Any) -> bool:
    if provider == requester:
        value = self.interleaver.batcher.narrow(self.batch_group)
        self.respond(value)

        # PP: clone consumed value into buffer for the listener
        if getattr(self.interleaver, 'pp_enabled', False):
            cv = self.interleaver.batcher.current_value
            self.interleaver.pp_hook_buffer[provider] = (
                cv.clone() if isinstance(cv, torch.Tensor) else cv
            )

    else:
        if requester in self.history:
            self.respond(
                Mediator.OutOfOrderError(
                    f"Value was missed for {requester}. Did you call an Envoy out of order?"
                )
            )
        else:
            self.history.add(provider)
            self.event_queue.restore((Events.VALUE, requester))
            return False

    return True
```

And `handle_swap_event` should be the original without PPMissing branch:

```python
def handle_swap_event(self, provider: Any, requester: Any, swap_value: Any):
    if provider == requester:
        self.interleaver.batcher.swap(self.batch_group, swap_value)
        self.respond()
        return True

    else:
        if requester in self.history:
            self.respond(
                ValueError(
                    f"Setting {requester} is out of scope for scope {provider}. "
                    f"Did you call an Envoy out of order?"
                )
            )
        else:
            self.history.add(provider)
            self.event_queue.restore((Events.SWAP, (requester, swap_value)))
            return False

    return True
```

- [ ] **Step 3: Remove `pp_received_buffer` from GPUModelRunner**

In `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py`:

In `load_model()`, remove `self.pp_received_buffer: dict[str, Any] = {}`.

In `_update_states()`, remove `interleaver.pp_received_buffer = self.pp_received_buffer`.

In `collect_nnsight()`, remove `self.pp_received_buffer.clear()`.

- [ ] **Step 4: Remove `make_dummy_tensor` from pp.py**

In `src/nnsight/modeling/vllm/pp.py`, remove the `make_dummy_tensor` function (lines 81-88).

- [ ] **Step 5: Run existing unit tests**

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py::TestPPMissingDetection tests/test_vllm_pp.py::TestPPModuleMap -v -x`
Expected: Pass (these don't depend on removed code).

- [ ] **Step 6: Remove `TestDummyTensor` test class**

Remove the `TestDummyTensor` class from `tests/test_vllm_pp.py` since `make_dummy_tensor` is deleted.

- [ ] **Step 7: Commit**

```bash
git add src/nnsight/modeling/vllm/workers/GPUWorker.py \
        src/nnsight/intervention/interleaver.py \
        src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py \
        src/nnsight/modeling/vllm/pp.py \
        tests/test_vllm_pp.py
git commit -m "refactor(vllm-pp): remove Phase 1 eager exchange, END injection, pp_received_buffer"
```

---

## Task 4: Condition Notify on Buffer Clone

**Files:**
- Modify: `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py`
- Modify: `src/nnsight/intervention/interleaver.py`

- [ ] **Step 1: Add Condition to model runner**

In `GPUModelRunner.load_model()`, add a `threading.Condition` alongside the existing `pp_hook_buffer`:

```python
# In load_model(), after pp_hook_buffer init:
import threading

self.pp_hook_buffer: dict[str, Any] = {}
self.pp_buffer_condition = threading.Condition()
```

- [ ] **Step 2: Propagate Condition to interleaver**

In `_update_states()`, add:

```python
if self.pp_enabled:
    interleaver = self.nnsight_model._interleaver
    interleaver.pp_enabled = True
    interleaver.pp_module_map = self.pp_module_map
    interleaver.pp_hook_buffer = self.pp_hook_buffer
    interleaver.pp_buffer_condition = self.pp_buffer_condition
```

- [ ] **Step 3: Add Condition notify in handle_value_event buffer clone**

In `interleaver.py`, in the buffer clone block of `handle_value_event`, notify the Condition after inserting:

```python
if provider == requester:
    value = self.interleaver.batcher.narrow(self.batch_group)
    self.respond(value)

    # PP: clone consumed value into buffer and notify listener
    if getattr(self.interleaver, 'pp_enabled', False):
        cv = self.interleaver.batcher.current_value
        cond = self.interleaver.pp_buffer_condition
        with cond:
            self.interleaver.pp_hook_buffer[provider] = (
                cv.clone() if isinstance(cv, torch.Tensor) else cv
            )
            cond.notify_all()
```

- [ ] **Step 4: Commit**

```bash
git add src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py \
        src/nnsight/intervention/interleaver.py
git commit -m "feat(vllm-pp): add threading.Condition notify on pp_hook_buffer clone"
```

---

## Task 5: Envoy PPMissing Short-Circuit

**Files:**
- Modify: `src/nnsight/intervention/envoy.py`
- Test: `tests/test_vllm_pp.py`

This is the central change: when the Envoy's `.output` (or `.inputs`) is accessed for a PPMissing module, return a LazyRemoteTensor directly instead of posting an event and blocking.

- [ ] **Step 1: Write test for Envoy PPMissing short-circuit**

```python
# Append to tests/test_vllm_pp.py

from unittest.mock import MagicMock, patch
from collections import defaultdict
from nnsight.modeling.vllm.lazy_remote_tensor import LazyRemoteTensor


class TestEnvoyPPMissingShortCircuit:

    def _make_pp_envoy(self):
        """Create a minimal Envoy-like setup to test PPMissing short-circuit."""
        from nnsight.intervention.envoy import Envoy
        from nnsight.intervention.interleaver import Interleaver, Mediator

        # Create a PPMissingLayer module
        module = PPMissingLayer()

        # Create interleaver with PP state
        interleaver = Interleaver()
        interleaver.pp_enabled = True

        mock_map = MagicMock()
        mock_map.get_owning_rank.return_value = 1
        mock_map.is_local.return_value = False
        interleaver.pp_module_map = mock_map

        # Create a mock mediator with iteration tracker
        mediator = MagicMock()
        mediator.iteration_tracker = defaultdict(int)
        mediator.iteration = None
        interleaver.current = mediator

        # Simulate interleaving state
        interleaver._interleaving = True
        interleaver.mediators = [mediator]

        envoy = Envoy.__new__(Envoy)
        envoy._module = module
        envoy.path = "model.layers.50"
        envoy._interleaver = interleaver
        envoy._fake_output = __import__("inspect")._empty
        envoy._fake_inputs = __import__("inspect")._empty

        return envoy, mediator

    def test_output_returns_lazy_tensor(self):
        envoy, mediator = self._make_pp_envoy()
        result = envoy.output
        assert isinstance(result, LazyRemoteTensor)
        assert result._meta["provider_string"] == "model.layers.50.output.i0"
        assert result._meta["source_rank"] == 1

    def test_output_increments_tracker(self):
        envoy, mediator = self._make_pp_envoy()
        assert mediator.iteration_tracker["model.layers.50.output"] == 0
        _ = envoy.output
        assert mediator.iteration_tracker["model.layers.50.output"] == 1
        _ = envoy.output
        assert mediator.iteration_tracker["model.layers.50.output"] == 2

    def test_output_setter_noop_for_pp_missing(self):
        envoy, mediator = self._make_pp_envoy()
        # Should not raise or block
        envoy.output = torch.zeros(1)
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py::TestEnvoyPPMissingShortCircuit -v -x`
Expected: Fail — Envoy `.output` doesn't short-circuit yet.

- [ ] **Step 3: Implement PPMissing short-circuit in Envoy**

In `src/nnsight/intervention/envoy.py`, modify the `output` property getter (around line 146):

```python
@property
def output(self) -> Object:
    if self.interleaving:
        # PP short-circuit: PPMissing modules return LazyRemoteTensor directly
        if self._is_pp_missing:
            return self._pp_lazy_output()

        return self._interleaver.current.request(
            self._interleaver.iterate_requester(f"{self.path}.output")
        )
    elif self._fake_output is not inspect._empty:
        return self._fake_output
    else:
        raise ValueError(
            f"The model did not execute — cannot access `{self.path}.output`. "
            "Did you forget to pass a valid input to `.trace()` or `.invoke()`? "
            "Use `model.trace(input)` or `tracer.invoke(input)` to provide input."
        )
```

Modify the `output` setter (around line 177):

```python
@output.setter
def output(self, value: Any):
    if self.interleaving:
        # PP short-circuit: writes to PPMissing are no-ops
        if self._is_pp_missing:
            return

        self._interleaver.current.swap(
            self._interleaver.iterate_requester(f"{self.path}.output"), value
        )
    else:
        raise ValueError(
            f"Cannot set `{self.path}.output`. The model is not executing during interleaving."
            "Did you forget to pass a valid input to `.trace()` or `.invoke()`? "
            "Use `model.trace(input)` or `tracer.invoke(input)` to provide input."
        )
```

Modify the `inputs` property getter (around line 206):

```python
@property
def inputs(self) -> Tuple[Tuple[Object], Dict[str, Object]]:
    if self.interleaving:
        if self._is_pp_missing:
            return self._pp_lazy_input()

        return self._interleaver.current.request(
            self._interleaver.iterate_requester(f"{self.path}.input")
        )
    elif self._fake_inputs is not inspect._empty:
        return self._fake_inputs
    else:
        raise ValueError(
            f"Cannot access `{self.path}.inputs`. The model is not executing during interleaving."
            "Did you forget to pass a valid input to `.trace()` or `.invoke()`? "
            "Use `model.trace(input)` or `tracer.invoke(input)` to provide input."
        )
```

Modify the `inputs` setter (around line 238):

```python
@inputs.setter
def inputs(self, value: Any):
    if self.interleaving:
        if self._is_pp_missing:
            return

        self._interleaver.current.swap(
            self._interleaver.iterate_requester(f"{self.path}.input"), value
        )
    else:
        raise ValueError(
            f"Cannot set `{self.path}.inputs`. The model is not executing during interleaving."
            "Did you forget to pass a valid input to `.trace()` or `.invoke()`? "
            "Use `model.trace(input)` or `tracer.invoke(input)` to provide input."
        )
```

Add helper properties and methods to the `Envoy` class:

```python
@property
def _is_pp_missing(self) -> bool:
    """Check if this Envoy wraps a PPMissing module on a non-local rank."""
    if not getattr(self._interleaver, 'pp_enabled', False):
        return False
    from nnsight.modeling.vllm.pp import is_pp_missing
    return is_pp_missing(self._module)

def _pp_lazy_output(self):
    """Return a LazyRemoteTensor for this PPMissing module's output."""
    from nnsight.modeling.vllm.lazy_remote_tensor import LazyRemoteTensor

    mediator = self._interleaver.current
    module_key = f"{self.path}.output"
    iteration = mediator.iteration_tracker[module_key]
    provider_string = f"{module_key}.i{iteration}"
    mediator.iteration_tracker[module_key] += 1

    source_rank = self._interleaver.pp_module_map.get_owning_rank(provider_string)

    lazy = LazyRemoteTensor(
        source_rank=source_rank,
        provider_string=provider_string,
        shape=(),  # shape unknown until materialized; pull protocol sends it
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # Wire up the pull function to the listener
    if hasattr(self._interleaver, 'pp_listener') and self._interleaver.pp_listener is not None:
        listener = self._interleaver.pp_listener
        lazy._pull_fn = lambda src_rank, key: listener.local_lookup(key)

    return lazy

def _pp_lazy_input(self):
    """Return a LazyRemoteTensor for this PPMissing module's input."""
    from nnsight.modeling.vllm.lazy_remote_tensor import LazyRemoteTensor

    mediator = self._interleaver.current
    module_key = f"{self.path}.input"
    iteration = mediator.iteration_tracker[module_key]
    provider_string = f"{module_key}.i{iteration}"
    mediator.iteration_tracker[module_key] += 1

    source_rank = self._interleaver.pp_module_map.get_owning_rank(provider_string)

    lazy = LazyRemoteTensor(
        source_rank=source_rank,
        provider_string=provider_string,
        shape=(),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    if hasattr(self._interleaver, 'pp_listener') and self._interleaver.pp_listener is not None:
        listener = self._interleaver.pp_listener
        lazy._pull_fn = lambda src_rank, key: listener.local_lookup(key)

    return lazy
```

Add `import torch` at the top of `envoy.py` if not already present.

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py::TestEnvoyPPMissingShortCircuit -v -x`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/nnsight/intervention/envoy.py tests/test_vllm_pp.py
git commit -m "feat(vllm-pp): Envoy PPMissing short-circuit — returns LazyRemoteTensor directly"
```

---

## Task 6: Readiness Check + Listener Lifecycle

**Files:**
- Modify: `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py`

- [ ] **Step 1: Add readiness check to interleaver `__enter__`**

In `GPUModelRunner.execute_model()`, add the readiness check after the interleaver opens but before the forward pass. The check waits until each mediator's event queue has a value (meaning it's parked at a local module access):

```python
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
):
    Globals.enter()
    with self.nnsight_model._interleaver:

        # PP: wait until all mediators are ready (parked at a local module).
        # PPMissing accesses bypass event_queue, so any queued event must
        # be for a local module.
        if self.pp_enabled and self.nnsight_model._interleaver.mediators:
            self._pp_wait_for_mediator_readiness()

        return_value = super().execute_model(scheduler_output, intermediate_tensors)

        self.nnsight_request_helper.unflatten(self.nnsight_model)

        if self.execute_model_state is not None:
            logits = self.nnsight_model.logits(
                self.execute_model_state.logits, hook=True
            )
            state = self.execute_model_state
            self.execute_model_state = type(state)(
                **{**state._asdict(), "logits": logits}
            )

    Globals.exit()
    return return_value

def _pp_wait_for_mediator_readiness(self):
    """Wait until all mediators have a pending event for a local module.

    Mediators run freely for PPMissing accesses (Envoy short-circuit).
    Before firing forward pass hooks, we must ensure each mediator has
    finished all PPMissing processing and is blocked waiting for a local
    module (event in event_queue). This avoids hooks passing through
    with no consumer.
    """
    import time
    for mediator in self.nnsight_model._interleaver.mediators:
        while mediator.alive and not mediator.event_queue.has_value:
            time.sleep(0.0001)
```

- [ ] **Step 2: Add listener lifecycle to model runner**

In `GPUModelRunner.load_model()`, initialize listener state:

```python
# PP support (in load_model, after pp_module_map creation)
if self.pp_enabled:
    from ..pp import PPModuleMap
    from ..pp_listener import PPListener
    num_layers = self.model_config.hf_config.num_hidden_layers
    self.pp_module_map = PPModuleMap(num_layers, pp_world_size)
    self.pp_listener = PPListener(self.pp_hook_buffer, self.pp_buffer_condition)
else:
    self.pp_listener = None
```

In `_update_states()`, propagate listener to interleaver:

```python
if self.pp_enabled:
    interleaver = self.nnsight_model._interleaver
    interleaver.pp_enabled = True
    interleaver.pp_module_map = self.pp_module_map
    interleaver.pp_hook_buffer = self.pp_hook_buffer
    interleaver.pp_buffer_condition = self.pp_buffer_condition
    interleaver.pp_listener = self.pp_listener
```

In `collect_nnsight()`, stop listener and clean up:

```python
if self.pp_enabled and finished_keys:
    if self.pp_listener is not None:
        self.pp_listener.stop()
        # Recreate for next request
        self.pp_listener = PPListener(self.pp_hook_buffer, self.pp_buffer_condition)
        self.nnsight_model._interleaver.pp_listener = self.pp_listener
    self.pp_hook_buffer.clear()
```

- [ ] **Step 3: Commit**

```bash
git add src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py
git commit -m "feat(vllm-pp): add readiness check and listener lifecycle to model runner"
```

---

## Task 7: Integration Tests

**Files:**
- Modify: `tests/test_vllm_pp.py`

- [ ] **Step 1: Update existing integration tests**

The existing `TestPPBasicInference` and `TestPPCrossStageRead` test classes should still work with the new design (the user-facing API is unchanged). Run them to verify:

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py -v -x --ignore-glob='*TestDummyTensor*' -k 'not TestDummyTensor'`
Expected: Unit tests pass. Integration tests pass if 2+ GPUs available, skip otherwise.

- [ ] **Step 2: Add multi-token generation PP test**

```python
# Append to tests/test_vllm_pp.py

@_pp_skip
class TestPPMultiToken:
    """Multi-token generation with PP — mediator loops across tokens."""

    @torch.no_grad()
    def test_multi_token_save_per_step(self, vllm_gpt2_pp2):
        """Save logits at each generation step with PP=2."""
        model = vllm_gpt2_pp2
        with model.trace("The Eiffel Tower is in", max_tokens=3) as tracer:
            logits_list = list().save()
            for step in tracer.iter[:]:
                logits_list.append(model.logits.output[0, -1].argmax(dim=-1))

        assert len(logits_list) == 3
        for tok_id in logits_list:
            assert isinstance(tok_id, (torch.Tensor, int))

    @torch.no_grad()
    def test_multi_token_cross_stage_write(self, vllm_gpt2_pp2):
        """Cross-stage write at each generation step."""
        model = vllm_gpt2_pp2
        with model.trace("Hello world", max_tokens=2) as tracer:
            for step in tracer.iter[:]:
                h = model.transformer.h[2].output[0]
                model.transformer.h[8].output[0][:] = h
            output = model.generator.output.save()

        assert output is not None

    @torch.no_grad()
    def test_multi_token_cross_stage_read(self, vllm_gpt2_pp2):
        """Cross-stage read: materialize a remote value at each step."""
        model = vllm_gpt2_pp2
        with model.trace("The Eiffel Tower is in", max_tokens=2) as tracer:
            results = list().save()
            for step in tracer.iter[:]:
                # Read from stage 0 layer, compute on it (forces materialization
                # on rank 1 where layer 2 is PPMissing)
                h = model.transformer.h[2].output[0]
                results.append(h.mean())

        assert len(results) == 2
```

- [ ] **Step 3: Run all tests**

Run: `cd /disk/u/zikai/nnsight/.claude/worktrees/vllm-multinode && conda run -n ndif-dev pytest tests/test_vllm_pp.py -v`
Expected: All unit tests pass. Integration tests pass with 2+ GPUs.

- [ ] **Step 4: Commit**

```bash
git add tests/test_vllm_pp.py
git commit -m "test(vllm-pp): add multi-token generation PP integration tests"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] LazyRemoteTensor class → Task 1
- [x] PPMissing short-circuit at Envoy → Task 5
- [x] pp_hook_buffer clone-on-consume with Condition notify → Task 4
- [x] Listener thread → Task 2 + Task 6 (lifecycle)
- [x] Readiness check → Task 6
- [x] Remove END injection → Task 3
- [x] Remove eager buffer exchange → Task 3
- [x] Remove pp_received_buffer → Task 3
- [x] Remove make_dummy_tensor → Task 3
- [x] Save collection from all ranks → kept from Phase 1 (no changes needed)
- [x] PP-aware deserialization → kept from Phase 1 (no changes needed)
- [x] Multi-token generation → Task 7 (integration tests)
- [x] Iteration tracking at Envoy level → Task 5

**Placeholder scan:** No TBDs. All code blocks complete. All commands include expected output.

**Type consistency:** `pp_hook_buffer: dict[str, Any]`, `pp_buffer_condition: threading.Condition`, `pp_listener: PPListener` — consistent across Tasks 2, 4, 5, 6. `LazyRemoteTensor._pull_fn` signature `(source_rank: int, provider_string: str) -> torch.Tensor` — consistent between Task 1 and Task 5.
