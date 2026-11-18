"""A Gemma tokenizer that leaves ``<bos>`` off a plain string is set to prepend it."""

import pytest
from transformers import (
    GemmaConfig,
    GemmaForCausalLM,
    GemmaTokenizer,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    SiglipVisionConfig,
)

import nnsight
from nnsight.modeling.transformers import TransformersModel

BOS = 2


def text_config() -> GemmaConfig:
    return GemmaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )


def load(module) -> TransformersModel:
    # With a tokenizer that leaves <bos> off, as the Gemma 4 -it ones do.
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": BOS, "<unk>": 3, "<mask>": 4}
    vocab.update({ch: i for i, ch in enumerate("▁helowrd", start=len(vocab))})
    tokenizer = GemmaTokenizer(vocab=vocab, merges=[], add_bos_token=False)
    tokenizer.chat_template = "{{ bos_token }}{{ messages[0]['content'] }}"
    return TransformersModel(module, task="text-generation", tokenizer=tokenizer, device="cpu")


@pytest.mark.parametrize("prompt", ["hello world", [{"role": "user", "content": "hello"}]])
def test_string_and_chat_start_with_one_bos(prompt):
    model = load(GemmaForCausalLM(text_config()))
    with model.trace(prompt):
        inputs = nnsight.save(model.inputs)
    ids = inputs[1]["input_ids"][0].tolist()
    assert ids[0] == BOS and ids.count(BOS) == 1


def test_paligemma_tokenizer_is_left_alone():
    # Its processor puts <bos> after the image tokens.
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
    )
    config = PaliGemmaConfig(text_config=text_config(), vision_config=vision)
    model = load(PaliGemmaForConditionalGeneration(config))
    assert model.tokenizer("hello").input_ids[0] != BOS
