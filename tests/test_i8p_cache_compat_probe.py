"""Tests for I8' (replacing the original I8 dim guard): startup
cache-compatibility probe in ``VanillaBatchServer``.

Background
----------
``VanillaBatchServer._activate_request`` hardcodes ``DynamicCache()``
for every request. Models that require a different cache class
(Mamba/RWKV via ``LinearAttentionLayer``; T5/BART via
``EncoderDecoderCache``; compile-targeted setups via ``StaticCache``)
crash somewhere deep in the model forward with a cryptic error
unrelated to the actual cause.

The probe runs a 1-token forward with a fresh ``DynamicCache`` once
at server startup. On failure, it raises a clear ``RuntimeError``
naming the model class, the underlying probe error, and the three
alternative backends (HF paged, vLLM serve, local generate).

The original I8 framing — a dim guard inside ``_merge_caches`` —
was based on the assumption that ``DynamicLayer.keys`` could have
non-4D shapes. Empirically false: the class is contractually 4D
and subclasses (``QuantizedLayer``, ``DynamicSlidingWindowLayer``)
preserve that shape. The actual failure mode for unsupported models
is cache-class mismatch, which manifests *before* ``_merge_caches``
runs. The probe is the right place to detect it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_probe_passes_on_dynamic_cache_compatible_model():
    """GPT-2 uses ``DynamicCache`` natively. Probe must succeed and
    not crash, signal, or warn.
    """
    from nnsight import LanguageModel
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    server = VanillaBatchServer(model, mediator_timeout=5.0)
    try:
        # ``start()`` runs the probe before launching the bg thread.
        server.start()
        assert server.is_running()
        assert getattr(server, "_probed", False) is True
    finally:
        server.stop()


def test_probe_raises_with_clear_message_when_model_rejects_dynamic_cache():
    """A model whose forward rejects ``DynamicCache`` (e.g. Mamba,
    RWKV, encoder-decoder) must surface as a clear ``RuntimeError``
    naming the model class, the underlying probe error, and the three
    alternative backends — not propagate the original opaque error.
    """
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    class _MambaLikeModel(torch.nn.Module):
        """Stand-in for an SSM model that can't use DynamicCache.
        Real Mamba would fail on accessing ``conv_states`` /
        ``recurrent_states`` from a DynamicLayer; mock that as a
        TypeError naming the cache class for realism.
        """

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kwargs):
            cache = kwargs.get("past_key_values")
            raise TypeError(
                f"Expected MambaCache, got {type(cache).__name__} — this "
                f"architecture uses LinearAttentionLayer, not DynamicLayer."
            )

    fake_model = _MambaLikeModel()
    fake_wrapper = SimpleNamespace(
        _model=fake_model,
        interleaver=SimpleNamespace(worker_context=None),
    )

    server = VanillaBatchServer.__new__(VanillaBatchServer)
    server.model = fake_wrapper
    server.worker_context = None
    # Bypass __init__ — we only need the bits ``_probe_cache_compatibility``
    # touches. Mark `_probed` False so the call actually runs.
    server._probed = False

    with pytest.raises(RuntimeError) as exc_info:
        server._probe_cache_compatibility()

    msg = str(exc_info.value)

    # Must name the model class so operators know which model they
    # tried to load.
    assert "_MambaLikeModel" in msg

    # Must include the underlying probe error so the cause is visible.
    assert "TypeError" in msg
    assert "MambaCache" in msg or "LinearAttentionLayer" in msg

    # Must point at the three alternative backends — operators need
    # to know how to recover.
    assert "NNsightCBManager" in msg or "paged" in msg.lower()
    assert "vLLM" in msg or "vllm" in msg
    assert "model.generate" in msg

    # Original exception preserved as ``__cause__`` — pytest debug
    # output and operator log readers can chain through.
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_probe_catches_silent_cache_type_mismatch_on_output():
    """A model whose forward ACCEPTS ``DynamicCache`` (because the
    kwarg is in the signature) but RETURNS a different cache class on
    ``outputs.past_key_values`` must be caught.

    This is Mamba's actual failure mode: ``MambaForCausalLM.forward``
    accepts ``past_key_values=DynamicCache()`` without complaint
    (it's permitted by the signature) but ignores it internally and
    returns a ``MambaCache`` instead. ``_split_cache`` would then
    crash later with an opaque error far from the actual cause.
    The probe's input-acceptance check alone wouldn't catch this —
    the round-trip check is what does.
    """
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    class _SilentlyIgnoresKwarg(torch.nn.Module):
        """Stand-in for Mamba: accepts past_key_values silently,
        returns no DynamicCache on the output."""

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kwargs):
            # Returns an object with logits and a NON-DynamicCache
            # for past_key_values (mimicking MambaCausalLMOutput).
            from types import SimpleNamespace

            class _NotADynamicCache:
                pass

            return SimpleNamespace(
                logits=torch.zeros(1, 1, 10),
                past_key_values=_NotADynamicCache(),
            )

    fake_model = _SilentlyIgnoresKwarg()
    fake_wrapper = SimpleNamespace(
        _model=fake_model,
        interleaver=SimpleNamespace(worker_context=None),
    )

    server = VanillaBatchServer.__new__(VanillaBatchServer)
    server.model = fake_wrapper
    server.worker_context = None
    server._probed = False

    with pytest.raises(RuntimeError) as exc_info:
        server._probe_cache_compatibility()

    msg = str(exc_info.value)

    # Must name the model class.
    assert "_SilentlyIgnoresKwarg" in msg
    # Must surface the type mismatch — not just "probe failed."
    assert "_NotADynamicCache" in msg
    assert "DynamicCache" in msg
    # Must point at the alternatives.
    assert "model.generate" in msg
    # No __cause__ for this branch — the model didn't raise; we did
    # after observing the bad return type.
    assert exc_info.value.__cause__ is None


def test_probe_accepts_none_past_key_values_as_failure():
    """Edge case: some custom architectures return outputs with
    ``past_key_values=None`` (the kwarg was accepted but ignored,
    no replacement was produced). Same failure shape as above —
    the probe must reject because ``_split_cache`` can't operate
    on ``None``.
    """
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    class _ReturnsNonePKV(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(
                logits=torch.zeros(1, 1, 10),
                past_key_values=None,
            )

    fake_wrapper = SimpleNamespace(
        _model=_ReturnsNonePKV(),
        interleaver=SimpleNamespace(worker_context=None),
    )
    server = VanillaBatchServer.__new__(VanillaBatchServer)
    server.model = fake_wrapper
    server.worker_context = None
    server._probed = False

    with pytest.raises(RuntimeError) as exc_info:
        server._probe_cache_compatibility()

    msg = str(exc_info.value)
    assert "NoneType" in msg
    assert "DynamicCache" in msg


def test_probe_rejects_vlm_with_vision_config():
    """LLaVA / Qwen2-VL / InternVL / Idefics / PaliGemma / BLIP-2 /
    Phi-3-Vision / DeepSeek-VL2 all expose ``config.vision_config``.
    Probe must detect and reject before the forward — even if the
    text decoder would pass the cache checks, batch construction
    doesn't carry pixel_values, so real image-bearing requests
    would silently corrupt.
    """
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    class _FakeVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)
            # Mimic LLaVA-style nested vision config.
            self.config = SimpleNamespace(
                vision_config=SimpleNamespace(hidden_size=1024),
            )

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kwargs):
            # If the probe gets here, the VLM check failed — text-only
            # forward would otherwise return a DynamicCache and the
            # probe would happily pass. Make this fail loudly so the
            # test surfaces the gap.
            raise AssertionError(
                "probe reached forward() — VLM detection didn't fire"
            )

    fake_wrapper = SimpleNamespace(
        _model=_FakeVLM(),
        interleaver=SimpleNamespace(worker_context=None),
    )
    server = VanillaBatchServer.__new__(VanillaBatchServer)
    server.model = fake_wrapper
    server.worker_context = None
    server._probed = False

    with pytest.raises(RuntimeError) as exc_info:
        server._probe_cache_compatibility()

    msg = str(exc_info.value)
    # Must name the model class.
    assert "_FakeVLM" in msg
    # Must explain the multimodal nature of the rejection.
    assert "vision-language" in msg.lower() or "multimodal" in msg.lower()
    # Must surface the actual concern: silent corruption from missing
    # pixel_values in batch construction.
    assert "pixel_values" in msg
    # Must point at the alternatives.
    assert "model.generate" in msg


def test_probe_rejects_vlm_with_image_token_id():
    """Some VLM configs only expose ``image_token_id`` (or
    ``image_token_index``) — the canonical config-key signal varies
    across HF versions and model families. Detection must catch
    these too."""
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    class _ImageTokenMarkerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)
            self.config = SimpleNamespace(image_token_id=32000)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kwargs):
            raise AssertionError("probe reached forward() — VLM check missed")

    fake_wrapper = SimpleNamespace(
        _model=_ImageTokenMarkerModel(),
        interleaver=SimpleNamespace(worker_context=None),
    )
    server = VanillaBatchServer.__new__(VanillaBatchServer)
    server.model = fake_wrapper
    server.worker_context = None
    server._probed = False

    with pytest.raises(RuntimeError) as exc_info:
        server._probe_cache_compatibility()
    assert "_ImageTokenMarkerModel" in str(exc_info.value)


def test_probe_rejects_vlm_with_vision_tower_attr():
    """Older LLaVA variants attach ``vision_tower`` directly to the
    model rather than nesting it in config. Detection must catch
    this case too."""
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    class _OldLLaVAStyle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)
            self.vision_tower = torch.nn.Linear(4, 4)  # not a real tower
            self.config = SimpleNamespace()  # no vision_config

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kwargs):
            raise AssertionError("probe reached forward() — VLM check missed")

    fake_wrapper = SimpleNamespace(
        _model=_OldLLaVAStyle(),
        interleaver=SimpleNamespace(worker_context=None),
    )
    server = VanillaBatchServer.__new__(VanillaBatchServer)
    server.model = fake_wrapper
    server.worker_context = None
    server._probed = False

    with pytest.raises(RuntimeError):
        server._probe_cache_compatibility()


def test_vlm_detection_does_not_false_positive_on_plain_decoder_only():
    """Plain decoder-only LMs (no vision config, no image token, no
    vision tower) must NOT be flagged as VLMs. Pin this so a future
    detection-signal addition doesn't accidentally widen the net to
    non-multimodal models.
    """
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    class _PlainDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=128, num_hidden_layers=4)

    assert VanillaBatchServer._is_vision_language_model(_PlainDecoder()) is False


@pytest.mark.parametrize("attr_path,attr_value", [
    ("config.vision_config", SimpleNamespace(hidden_size=1024)),
    ("config.image_token_id", 32000),
    ("config.image_token_index", 32000),
])
def test_vlm_detection_isolated_signals(attr_path, attr_value):
    """Each detection signal in isolation must fire. Parametrized so
    a future signal addition gets the same shape of test for free.
    Module-level attributes (vision_tower / vision_model) are covered
    by the dedicated test above since they require nn.Module shapes.
    """
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    parts = attr_path.split(".")
    if parts[0] == "config":
        config = SimpleNamespace()
        setattr(config, parts[1], attr_value)
        model = SimpleNamespace(config=config)
    else:
        raise AssertionError(f"unhandled attr_path in test: {attr_path}")

    assert VanillaBatchServer._is_vision_language_model(model) is True


def test_probe_runs_once_across_stop_start_cycles():
    """``_probed`` flag avoids re-probing on stop/restart. Probing is
    cheap (1-token forward) but pointless to repeat — same model.
    """
    from nnsight import LanguageModel
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    server = VanillaBatchServer(model, mediator_timeout=5.0)

    # First start: probe runs.
    probe_calls = {"n": 0}
    original_probe = server._probe_cache_compatibility

    def counting_probe():
        probe_calls["n"] += 1
        return original_probe()

    server._probe_cache_compatibility = counting_probe

    try:
        server.start()
        assert probe_calls["n"] == 1
        server.stop()
        # Second start: probe must NOT run again.
        server.start()
        assert probe_calls["n"] == 1, (
            f"probe ran {probe_calls['n']} times across stop/start "
            f"cycles; expected 1"
        )
    finally:
        server.stop()


def test_probe_does_not_leak_into_first_real_request():
    """The probe runs a forward with input_ids=[[0]]. That forward
    must NOT leave the model in a state that contaminates the first
    real client request. Smoke-check by running a real trace right
    after start() and verifying the saves come back correct.
    """
    from nnsight import LanguageModel
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    server = VanillaBatchServer(model, mediator_timeout=10.0)
    try:
        server.start()
        # Smallest possible end-to-end exercise — same shape as
        # test_hf_serve.py uses for sanity checks.
        with model.trace("Hello", server=server, max_new_tokens=1):
            saved = model.lm_head.output.save()
        # nnsight 0.5+: ``saved`` IS the tensor after the trace exits.
        assert saved is not None
        assert saved.shape[-1] == model.config.vocab_size
    finally:
        server.stop()
