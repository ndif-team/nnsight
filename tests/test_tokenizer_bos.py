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


def load(module, add_bos_token: bool = False) -> TransformersModel:
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": BOS, "<unk>": 3, "<mask>": 4}
    vocab.update({ch: i for i, ch in enumerate("▁helowrd", start=len(vocab))})
    tokenizer = GemmaTokenizer(vocab=vocab, merges=[], add_bos_token=add_bos_token)
    tokenizer.chat_template = "{{ bos_token }}{{ messages[0]['content'] }}"
    return TransformersModel(module, task="text-generation", tokenizer=tokenizer, device="cpu")


@pytest.mark.parametrize("add_bos_token", [False, True])
@pytest.mark.parametrize("prompt", ["hello world", [{"role": "user", "content": "hello"}]])
def test_string_and_chat_start_with_one_bos(add_bos_token, prompt):
    model = load(GemmaForCausalLM(text_config()), add_bos_token)
    with model.trace(prompt):
        inputs = nnsight.save(model.inputs)
    ids = inputs[1]["input_ids"][0].tolist()
    assert ids[0] == BOS and ids.count(BOS) == 1


def test_image_and_text_start_with_one_bos():
    image = pytest.importorskip("PIL.Image").new("RGB", (64, 64))
    repo = "trl-internal-testing/tiny-Gemma4ForConditionalGeneration"
    processor = TransformersModel(repo, task="image-text-to-text").processor
    text = processor.image_token + "caption"
    ids = processor(text=text, images=image, return_tensors="pt")["input_ids"][0].tolist()
    assert ids[0] == processor.tokenizer.bos_token_id and ids.count(ids[0]) == 1


def test_paligemma_tokenizer_is_left_alone():
    # Its processor puts <bos> after the image tokens.
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
    )
    config = PaliGemmaConfig(text_config=text_config(), vision_config=vision)
    model = load(PaliGemmaForConditionalGeneration(config))
    assert model.tokenizer("hello").input_ids[0] != BOS


def test_gpt2_declares_a_bos_and_still_encodes_without_it(gpt2):
    assert gpt2.tokenizer("hello").input_ids[0] != gpt2.tokenizer.bos_token_id
