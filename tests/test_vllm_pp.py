import inspect
import threading
import time
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import torch

from nnsight.modeling.vllm.lazy_remote_tensor import LazyRemoteTensor
from nnsight.modeling.vllm.pp_listener import PPListener
from vllm.model_executor.models.utils import PPMissingLayer


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
        envoy._fake_output = inspect._empty
        envoy._fake_inputs = inspect._empty

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
