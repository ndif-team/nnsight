"""Encoder-decoder models (BART) pad on the right.

BART's encoder reads learned absolute positions, so a left-padded row is shifted
and its batched result differs from the same row run alone. The text-generation
pipeline an encoder-decoder loads under left-pads whatever it wraps, so the side is
set from the model.
"""

import pytest
import torch
from transformers import BartConfig, BartForConditionalGeneration, BertTokenizer

import nnsight
from nnsight.modeling.transformers import TransformersModel

LONG = [2, 8, 9, 10, 11, 12, 13, 6, 7, 3]
SHORT = [2, 5, 6, 3]
START = torch.tensor([[3]])


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nother\nthe\nquick\nbrown\nfox\njumps\nover\n"
    )
    return BertTokenizer(vocab_file=str(vocab))


@pytest.fixture
def bart():
    # A large init_std makes the learned positions matter, which a default-scale
    # random model would hide.
    torch.manual_seed(7)
    config = BartConfig(
        vocab_size=14,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        decoder_start_token_id=3,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.5,
    )
    return BartForConditionalGeneration(config).eval()


@pytest.fixture
def model(bart, tokenizer):
    return TransformersModel(bart, task="text-generation", tokenizer=tokenizer, device="cpu")


def ids(tokens):
    return torch.tensor([tokens])


class TestEncoderDecoderPadding:
    def test_pads_right(self, model):
        assert model._is_causal() is False
        assert model.tokenizer.padding_side == "right"

    @torch.no_grad()
    def test_short_row_matches_the_row_alone(self, model, bart):
        with model.trace() as tracer:
            with tracer.invoke(ids(LONG), decoder_input_ids=START):
                pass
            with tracer.invoke(ids(SHORT), decoder_input_ids=START):
                logits = nnsight.save(model.output.logits)

        alone = bart(input_ids=ids(SHORT), decoder_input_ids=START).logits
        torch.testing.assert_close(logits[0], alone[0], atol=1e-5, rtol=1e-5)

    @torch.no_grad()
    def test_batched_generation_matches_the_row_alone(self, model, bart):
        with model.generate(max_new_tokens=4, do_sample=False) as tracer:
            with tracer.invoke(ids(LONG)):
                pass
            with tracer.invoke(ids(SHORT)):
                generated = nnsight.save(tracer.result)

        alone = bart.generate(ids(SHORT), max_new_tokens=4, do_sample=False)
        row = generated[0][generated[0] != bart.config.pad_token_id]
        assert row.tolist() == alone[0][alone[0] != bart.config.pad_token_id].tolist()
