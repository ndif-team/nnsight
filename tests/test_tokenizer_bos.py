"""A Gemma tokenizer that leaves ``<bos>`` off a plain string is set to prepend it.

Everything here is built from a tiny vocabulary, so no test touches the Hub.
"""

import pytest
import torch
from transformers import (
    GemmaConfig,
    GemmaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers.models.gemma.tokenization_gemma import GemmaTokenizer

import nnsight
from nnsight.modeling.transformers import TransformersModel

BOS = 2
PROMPT = "hello world"
CHAT = [{"role": "user", "content": PROMPT}]
TEMPLATE = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"


def gemma_tokenizer(add_bos_token: bool) -> GemmaTokenizer:
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": BOS, "<unk>": 3, "<mask>": 4}
    vocab.update({ch: i for i, ch in enumerate("▁helowrd", start=len(vocab))})
    tokenizer = GemmaTokenizer(vocab=vocab, merges=[], add_bos_token=add_bos_token)
    tokenizer.chat_template = TEMPLATE
    return tokenizer


def gemma_module() -> GemmaForCausalLM:
    torch.manual_seed(7)
    config = GemmaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    return GemmaForCausalLM(config).eval()


def gemma_model(tokenizer) -> TransformersModel:
    return TransformersModel(
        gemma_module(), task="text-generation", tokenizer=tokenizer, device="cpu"
    )


def traced_ids(model: TransformersModel, prompt) -> torch.Tensor:
    with model.trace(prompt):
        inputs = nnsight.save(model.inputs)
    return inputs[1]["input_ids"]


@pytest.fixture
def bare_tokenizer():
    tokenizer = gemma_tokenizer(add_bos_token=False)
    assert tokenizer(PROMPT)["input_ids"][0] != BOS
    return tokenizer


class TestGemma:
    def test_string_prompt_starts_with_bos(self, bare_tokenizer):
        model = gemma_model(bare_tokenizer)
        text = bare_tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
        ids = traced_ids(model, PROMPT)
        assert model.tokenizer.add_bos_token
        assert ids.tolist() == [[BOS, *text]]

    def test_chat_keeps_the_template_bos_only(self, bare_tokenizer):
        ids = traced_ids(gemma_model(bare_tokenizer), CHAT)
        assert ids[0, 0] == BOS
        assert (ids == BOS).sum() == 1

    @pytest.mark.parametrize("ids", [[BOS, 5, 6], [5, 6]])
    def test_pretokenized_ids_are_taken_as_given(self, bare_tokenizer, ids):
        assert traced_ids(gemma_model(bare_tokenizer), ids).tolist() == [ids]

    def test_tokenizer_that_already_prepends_is_untouched(self):
        tokenizer = gemma_tokenizer(add_bos_token=True)
        ids = traced_ids(gemma_model(tokenizer), PROMPT)
        assert ids[0, 0] == BOS
        assert (ids == BOS).sum() == 1

    def test_nested_text_config_is_read(self, bare_tokenizer):
        gemma4 = pytest.importorskip("transformers.models.gemma4")
        text = gemma4.Gemma4TextConfig(
            vocab_size=16,
            vocab_size_per_layer_input=16,
            hidden_size=16,
            hidden_size_per_layer_input=8,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            sliding_window=16,
        )
        vision = gemma4.Gemma4VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            patch_size=8,
        )
        audio = gemma4.Gemma4AudioConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            subsampling_conv_channels=[4, 4],
        )
        config = gemma4.Gemma4Config(text_config=text, vision_config=vision, audio_config=audio)
        assert not hasattr(config, "bos_token_id")
        torch.manual_seed(7)
        module = gemma4.Gemma4ForConditionalGeneration(config).eval()
        model = TransformersModel(
            module, task="text-generation", tokenizer=bare_tokenizer, device="cpu"
        )
        assert traced_ids(model, PROMPT)[0, 0] == BOS

    def test_tokenizer_the_setting_does_not_fix_warns(self):
        class Stubborn(GemmaTokenizer):
            add_bos_token = property(lambda self: False, lambda self, value: None)

        vocab = {"<pad>": 0, "<eos>": 1, "<bos>": BOS, "<unk>": 3, "<mask>": 4}
        tokenizer = Stubborn(vocab=vocab, merges=[])
        with pytest.warns(UserWarning, match="does not prepend <bos>"):
            gemma_model(tokenizer)


class TestGPT2:
    def test_encoding_is_unchanged(self):
        vocab = {"<|endoftext|>": 0}
        vocab.update({ch: i for i, ch in enumerate("Ġhelowrd", start=1)})
        tokenizer = GPT2Tokenizer(vocab=vocab, merges=[])
        assert tokenizer.bos_token_id == 0
        assert tokenizer(PROMPT)["input_ids"][0] != 0
        torch.manual_seed(7)
        config = GPT2Config(
            vocab_size=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=64,
            bos_token_id=0,
            eos_token_id=0,
        )
        model = TransformersModel(
            GPT2LMHeadModel(config).eval(),
            task="text-generation",
            tokenizer=tokenizer,
            device="cpu",
        )
        assert traced_ids(model, PROMPT).tolist() == [tokenizer(PROMPT)["input_ids"]]
        assert not model.tokenizer.add_bos_token
