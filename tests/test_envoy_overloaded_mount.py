"""Encoder models cannot be executed remotely: the Envoy class they get is unpicklable.

`Envoy._handle_overloaded_mount` fires whenever a wrapped module already defines
an attribute nnsight wants to mount -- in practice `.output`, which every
HuggingFace BERT- and ESM-family layer defines as a submodule
(`BertLayer.output = BertOutput(...)`). It resolves the collision by synthesizing
a subclass at runtime:

    new_cls = type(f"{self.__class__.__name__}.Preserved", (self.__class__,), {})
    object.__setattr__(self, "__class__", new_cls)

The synthesized class has a dot in `__name__`, is never bound in its defining
module, and so cannot be looked up by qualified name. Pickling it fails, and
remote execution of any encoder model fails on NDIF with
`RemoteException: name 'hooked_output' is not defined`.

Decoder models never enter this path, which is why GPT-2, Pythia and Llama work
remotely and BERT and ESM do not.

The local tests need no NDIF access and run on tiny models in seconds. The remote
tests are skipped unless NDIF_KEY is set.

    pytest test_envoy_overloaded_mount.py -v
"""

import os
import pickle

import pytest

from nnsight import Envoy, LanguageModel

# Tiny models so the local tests are CI-cheap. Both are public.
TINY_DECODER = "sshleifer/tiny-gpt2"
TINY_ENCODER = "hf-internal-testing/tiny-random-BertModel"

needs_ndif = pytest.mark.skipif(
    not os.environ.get("NDIF_KEY"), reason="NDIF_KEY not set"
)


@pytest.fixture(scope="module")
def decoder():
    return LanguageModel(TINY_DECODER)


@pytest.fixture(scope="module")
def encoder():
    from transformers import AutoModelForMaskedLM

    return LanguageModel(TINY_ENCODER, automodel=AutoModelForMaskedLM)


# ── local: no cluster required ───────────────────────────────────────────────


def test_decoder_layer_envoy_keeps_the_base_class(decoder):
    assert type(decoder.transformer.h[0]) is Envoy


def test_decoder_layer_envoy_class_is_picklable(decoder):
    pickle.dumps(type(decoder.transformer.h[0]))


def test_encoder_layer_envoy_class_name_has_no_dot(encoder):
    """A dot in __name__ makes the class unresolvable by qualified name."""
    cls = type(encoder.bert.encoder.layer[0])
    assert "." not in cls.__name__, (
        f"synthesized class is named {cls.__name__!r}; a dot in __name__ means "
        "pickle and importlib cannot resolve it"
    )


def test_encoder_layer_envoy_class_is_resolvable_in_its_module(encoder):
    import importlib

    cls = type(encoder.bert.encoder.layer[0])
    module = importlib.import_module(cls.__module__)
    assert getattr(module, cls.__name__, None) is cls, (
        f"{cls.__module__}.{cls.__name__} does not resolve back to the class"
    )


def test_encoder_layer_envoy_class_is_picklable(encoder):
    """The failing test. Passes for decoders, fails for every encoder."""
    pickle.dumps(type(encoder.bert.encoder.layer[0]))


def test_remapped_accessor_is_present(encoder):
    """Guards the intended behavior of the collision handling itself.

    Whatever fix lands must keep `.nns_output` mounted, so this should pass
    before and after.
    """
    assert hasattr(type(encoder.bert.encoder.layer[0]), "nns_output")


# ── remote: needs NDIF_KEY ───────────────────────────────────────────────────


@needs_ndif
def test_remote_decoder_returns_activations():
    lm = LanguageModel("EleutherAI/pythia-160m")
    with lm.trace("The capital of France is", remote=True):
        h = lm.gpt_neox.layers[6].output[0].save()
    assert h.shape[-1] == 768


@needs_ndif
def test_remote_encoder_returns_activations():
    """Fails with: RemoteException: name 'hooked_output' is not defined."""
    bm = LanguageModel("google-bert/bert-base-uncased")
    with bm.trace("The capital of France is [MASK].", remote=True):
        h = bm.bert.encoder.layer[6].nns_output[0].save()
    assert h.shape[-1] == 768
