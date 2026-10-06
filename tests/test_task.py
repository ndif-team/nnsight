"""The task a ``TransformersModel`` picks when ``task=`` is not given.

It comes from the checkpoint's config and transformers' auto mappings, never the
Hub's ``pipeline_tag``: every test here fails if anything asks the Hub for a
task, so a load with no task works offline and from a local directory.
"""

import glob
import json
import os
import shutil

import pytest

from nnsight.modeling.transformers import TransformersModel

GPT2 = "gpt2"
GEMMA3 = "yujiepan/gemma-3-tiny-random"
LLAMA4 = "yujiepan/llama-4-tiny-random"


@pytest.fixture(autouse=True)
def no_hub_task(monkeypatch):
    """Any lookup of the repo's pipeline tag fails the test."""
    import transformers.pipelines

    def refuse(*args, **kwargs):
        raise AssertionError("asked the Hub for a task")

    monkeypatch.setattr(transformers.pipelines, "get_task", refuse)


def _snapshot(repo):
    pattern = f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"
    snapshots = glob.glob(os.path.expanduser(pattern))
    if not snapshots:
        pytest.skip(f"{repo} is not cached")
    return snapshots[0]


def _local_copy(repo, directory, edit=None):
    """The cached checkpoint copied into ``directory``, its config optionally edited."""
    snapshot = _snapshot(repo)
    for name in os.listdir(snapshot):
        shutil.copy(os.path.realpath(os.path.join(snapshot, name)), directory / name)
    if edit is not None:
        path = directory / "config.json"
        config = json.loads(path.read_text())
        edit(config)
        path.write_text(json.dumps(config))
    return str(directory)


@pytest.fixture(scope="module")
def llama4(tmp_path_factory):
    """The tiny Llama 4 with ``attn_temperature_tuning`` as the bool transformers validates."""

    def as_bool(config):
        text = config["text_config"]
        text["attn_temperature_tuning"] = bool(text["attn_temperature_tuning"])

    return _local_copy(LLAMA4, tmp_path_factory.mktemp("llama4"), as_bool)


def test_text_model_is_text_generation():
    model = TransformersModel(GPT2)
    assert model.task == "text-generation"
    assert model.tokenizer is not None and model.processor is None


def test_multimodal_wrapper_is_image_text_to_text():
    pytest.importorskip("PIL")
    model = TransformersModel(GEMMA3, dispatch=True)
    assert model.task == "image-text-to-text"
    assert type(model._module).__name__ == "Gemma3ForConditionalGeneration"
    assert model.processor is not None
    assert model.tokenizer is model.processor.tokenizer
    with model.trace("text only"):
        logits = model.output.logits.save()
    assert logits.shape[:2] == (1, len(model.tokenizer("text only").input_ids))


def test_both_classes_config_defaults_to_the_wrapper(llama4):
    pytest.importorskip("PIL")
    model = TransformersModel(llama4)
    assert model.task == "image-text-to-text"
    assert type(model._module).__name__ == "Llama4ForConditionalGeneration"
    assert model.processor is not None


def test_explicit_task_wins(llama4):
    model = TransformersModel(llama4, task="text-generation")
    assert model.task == "text-generation"
    assert type(model._module).__name__ == "Llama4ForCausalLM"
    assert model.tokenizer is not None and model.processor is None


def test_local_directory(tmp_path):
    path = _local_copy("hf-internal-testing/tiny-random-LlamaForCausalLM", tmp_path)
    model = TransformersModel(path, dispatch=True)
    assert model.task == "text-generation"
    with model.trace("hello"):
        logits = model.output.logits.save()
    assert logits.shape[0] == 1


@pytest.mark.parametrize(
    "repo, task",
    [
        ("google-bert/bert-base-uncased", "fill-mask"),
        ("hf-internal-testing/tiny-random-WhisperForConditionalGeneration", "automatic-speech-recognition"),
    ],
)
def test_other_architectures_find_their_pipeline(repo, task):
    _snapshot(repo)
    assert TransformersModel(repo).task == task


def test_unknown_architecture_asks_for_a_task():
    from transformers import T5Config

    from nnsight.modeling.transformers import _task_from_config

    with pytest.raises(ValueError, match="T5Config.*task="):
        _task_from_config(T5Config(architectures=["T5ForConditionalGeneration"]))


def test_preloaded_wrapper_gets_a_processor():
    pytest.importorskip("PIL")
    from transformers import AutoModelForImageTextToText

    module = AutoModelForImageTextToText.from_pretrained(GEMMA3)
    model = TransformersModel(module)
    assert model.task == "image-text-to-text"
    assert model.processor is not None
    assert model.tokenizer is model.processor.tokenizer


class TestRemoteKey:
    """The key names a task only when it builds a class the default would not."""

    @staticmethod
    def _key(model):
        from nnsight.modeling.huggingface import _ID_CACHE

        _ID_CACHE.setdefault(model.repo_id, model.repo_id)
        return json.loads(model.to_model_key().split(":", 1)[1])

    def test_default_task_is_not_keyed(self, llama4):
        assert "task" not in self._key(TransformersModel(llama4))

    def test_class_changing_task_is_keyed_and_rebuilt(self, llama4):
        from nnsight.modeling.mixins.remotable import Remotable

        model = TransformersModel(llama4, task="text-generation")
        assert self._key(model)["task"] == "text-generation"
        server = Remotable.from_model_key(model.to_model_key())
        assert type(server._module) is type(model._module)

    def test_same_class_task_is_not_keyed(self):
        # Gemma 3's causal-LM class is its wrapper, so the key stays the same.
        model = TransformersModel(GEMMA3, task="text-generation")
        assert "task" not in self._key(model)
