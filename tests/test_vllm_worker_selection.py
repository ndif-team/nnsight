"""Exercise runner injection without importing CUDA-dependent vLLM modules."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def worker_runtime(monkeypatch):
    package = "_nnsight_worker_selection"
    names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu_worker",
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.executor",
        "vllm.v1.executor.ray_executor",
        package,
        f"{package}.workers",
        f"{package}.model_runners",
        f"{package}.model_runners.GPUModelRunner",
        f"{package}.model_runners.GPUModelRunnerV2",
    ]
    modules = {}
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)

    class OriginalRunner:
        def __init__(self, config, device):
            self.config = config

        def collect_nnsight(self, req_ids, finished_req_ids):
            return req_ids, finished_req_ids

    class OriginalRunnerV2(OriginalRunner):
        pass

    class NNsightGPUModelRunner(OriginalRunner):
        pass

    class NNsightGPUModelRunnerV2(OriginalRunnerV2):
        @staticmethod
        def validate_config(config):
            if config.is_mm_encoder_only:
                raise ValueError("NNsight does not support encoder-only runners")

    class RayDistributedExecutor:
        pass

    modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = OriginalRunner
    modules["vllm.v1.worker.gpu.model_runner"].GPUModelRunner = OriginalRunnerV2
    modules[
        f"{package}.model_runners.GPUModelRunner"
    ].NNsightGPUModelRunner = NNsightGPUModelRunner
    modules[
        f"{package}.model_runners.GPUModelRunnerV2"
    ].NNsightGPUModelRunnerV2 = NNsightGPUModelRunnerV2
    modules[
        "vllm.v1.executor.ray_executor"
    ].RayDistributedExecutor = RayDistributedExecutor

    calls = []

    class Worker:
        def __init__(self, use_v2, backend=None, fail=False, bypass=False):
            self.use_v2_model_runner = use_v2
            self.parallel_config = SimpleNamespace(distributed_executor_backend=backend)
            self.vllm_config = SimpleNamespace(is_mm_encoder_only=False)
            self.fail = fail
            self.bypass = bypass

        def init_device(self):
            calls.append(self.parallel_config.distributed_executor_backend)
            if self.fail:
                raise RuntimeError("device initialization failed")
            if self.use_v2_model_runner:
                from vllm.v1.worker.gpu.model_runner import GPUModelRunner
            else:
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner
            if self.bypass:
                GPUModelRunner = OriginalRunner
            self.model_runner = GPUModelRunner(self.vllm_config, "cuda")

    modules["vllm.v1.worker.gpu_worker"].Worker = Worker
    worker_path = (
        Path(__file__).parents[1]
        / "src/nnsight/modeling/vllm/workers/GPUWorker.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"{package}.workers.GPUWorker", worker_path
    )
    worker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)
    return SimpleNamespace(
        worker_cls=worker_module.NNsightGPUWorker,
        upstream_worker_cls=Worker,
        modules=modules,
        package=package,
        adapters=(NNsightGPUModelRunner, NNsightGPUModelRunnerV2),
        originals=(OriginalRunner, OriginalRunnerV2),
        ray_cls=RayDistributedExecutor,
        calls=calls,
    )


@pytest.mark.parametrize("use_v2", [False, True])
def test_selected_runner_is_scoped_and_unselected_adapter_is_not_imported(
    worker_runtime, monkeypatch, use_v2
):
    runtime = worker_runtime
    # An incompatible or absent unselected adapter must not prevent startup.
    unselected = "GPUModelRunner" if use_v2 else "GPUModelRunnerV2"
    monkeypatch.setitem(
        sys.modules, f"{runtime.package}.model_runners.{unselected}", None
    )
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0" if use_v2 else "1")
    worker = runtime.worker_cls(use_v2)
    worker.init_device()

    assert isinstance(worker.model_runner, runtime.adapters[use_v2])
    assert (
        runtime.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner
        is runtime.originals[0]
    )
    assert (
        runtime.modules["vllm.v1.worker.gpu.model_runner"].GPUModelRunner
        is runtime.originals[1]
    )
    native_worker = runtime.upstream_worker_cls(use_v2)
    native_worker.init_device()
    assert type(native_worker.model_runner) is runtime.originals[use_v2]
    assert worker.collect_nnsight(["request"], ["finished"]) == (
        ["request"],
        ["finished"],
    )


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_runner_and_ray_backend_are_restored(worker_runtime, use_v2, fail):
    runtime = worker_runtime

    class NNsightRayExecutor(runtime.ray_cls):
        pass

    worker = runtime.worker_cls(use_v2, backend=NNsightRayExecutor, fail=fail)
    if fail:
        with pytest.raises(RuntimeError, match="device initialization failed"):
            worker.init_device()
    else:
        worker.init_device()

    assert runtime.calls == ["ray"]
    assert worker.parallel_config.distributed_executor_backend is NNsightRayExecutor
    assert (
        runtime.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner
        is runtime.originals[0]
    )
    assert (
        runtime.modules["vllm.v1.worker.gpu.model_runner"].GPUModelRunner
        is runtime.originals[1]
    )


@pytest.mark.parametrize("use_v2", [False, True])
def test_worker_rejects_silent_runner_bypass(worker_runtime, use_v2):
    worker = worker_runtime.worker_cls(use_v2, bypass=True)
    with pytest.raises(RuntimeError, match="did not initialize.*NNsight model runner"):
        worker.init_device()


def test_v2_config_is_validated_before_initializing_device(worker_runtime):
    worker = worker_runtime.worker_cls(True)
    worker.vllm_config.is_mm_encoder_only = True
    with pytest.raises(ValueError, match="encoder-only"):
        worker.init_device()
    assert worker_runtime.calls == []
