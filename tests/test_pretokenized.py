"""Pretokenized traces preserve the model inputs produced by a tokenizer."""

import pytest
import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
)

import nnsight
from nnsight.modeling.transformers import TransformersModel


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nother\n")
    return BertTokenizer(vocab_file=str(vocab))


def tiny_config():
    return BertConfig(
        vocab_size=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(7)
    module = BertForMaskedLM(tiny_config()).eval()
    return TransformersModel(module, task="fill-mask", tokenizer=tokenizer, device="cpu")


@pytest.mark.parametrize("form", ["encoding", "keywords", "token_ids"])
@pytest.mark.parametrize("field", ["token_type_ids", "position_ids", "labels"])
@torch.no_grad()
def test_trace_preserves_each_model_input(model, form, field):
    inputs = {
        "input_ids": torch.tensor([[2, 5, 4, 3]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        field: {
            "token_type_ids": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.tensor([[4, 5, 6, 7]]),
            "labels": torch.tensor([[-100, -100, 6, -100]]),
        }[field],
    }
    expected = model._module(**inputs)
    if form == "encoding":
        with model.trace(inputs):
            actual = nnsight.save(model.output)
    elif form == "keywords":
        with model.trace(**inputs):
            actual = nnsight.save(model.output)
    else:
        ids = inputs.pop("input_ids")
        with model.trace(ids, **inputs):
            actual = nnsight.save(model.output)
    torch.testing.assert_close(actual.logits, expected.logits)
    if field == "labels":
        torch.testing.assert_close(actual.loss, expected.loss)


@torch.no_grad()
def test_tokenizer_sentence_pair_matches_native_forward(model):
    encoding = model.tokenizer("hello", "world", return_tensors="pt")
    assert encoding.token_type_ids.any()
    expected = model._module(**encoding)
    with model.trace(encoding):
        actual = nnsight.save(model.output.logits)
    torch.testing.assert_close(actual, expected.logits)


@pytest.mark.parametrize("padding_side", ["left", "right"])
@torch.no_grad()
def test_unequal_invokes_preserve_positions_and_ignore_added_label_padding(
    model, padding_side
):
    model.tokenizer.padding_side = padding_side
    short = {
        "input_ids": torch.tensor([[2, 4, 3]]),
        "token_type_ids": torch.tensor([[0, 1, 1]]),
        "position_ids": torch.tensor([[5, 6, 7]]),
        "labels": torch.tensor([[-100, 6, -100]]),
    }
    long = {
        "input_ids": torch.tensor([[2, 5, 4, 3]]),
        "token_type_ids": torch.tensor([[0, 0, 1, 1]]),
        "position_ids": torch.tensor([[2, 3, 4, 5]]),
        "labels": torch.tensor([[-100, -100, 7, -100]]),
    }
    padding = (1, 0) if padding_side == "left" else (0, 1)
    expected_inputs = {
        key: torch.cat(
            (torch.nn.functional.pad(short[key], padding, value=fill), long[key])
        )
        for key, fill in {
            "input_ids": 0,
            "token_type_ids": 0,
            "position_ids": 0,
            "labels": -100,
        }.items()
    }
    expected_inputs["attention_mask"] = torch.cat(
        (
            torch.nn.functional.pad(torch.ones_like(short["input_ids"]), padding),
            torch.ones_like(long["input_ids"]),
        )
    )
    expected = model._module(**expected_inputs)
    with model.trace() as tracer:
        with tracer.invoke(short):
            pass
        with tracer.invoke(long):
            pass
        with tracer.invoke():
            received = nnsight.save(model.inputs[1])
            actual = nnsight.save(model.output)
    for key, value in expected_inputs.items():
        torch.testing.assert_close(received[key], value)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.loss, expected.loss)


@torch.no_grad()
def test_sequence_classification_labels_keep_the_batch_shape(tokenizer):
    module = BertForSequenceClassification(tiny_config()).eval()
    model = TransformersModel(
        module, task="text-classification", tokenizer=tokenizer, device="cpu"
    )
    encoding = tokenizer(["hello", "hello world"], padding=True, return_tensors="pt")
    encoding["labels"] = torch.tensor([0, 1])
    expected = module(**encoding)
    with model.trace(encoding):
        received = nnsight.save(model.inputs[1]["labels"])
        actual = nnsight.save(model.output)
    torch.testing.assert_close(received, encoding["labels"])
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.loss, expected.loss)


@torch.no_grad()
def test_left_padding_still_supplies_positions_when_omitted(model):
    model.tokenizer.padding_side = "left"
    encoding = {
        "input_ids": torch.tensor([[0, 2, 4, 3]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1]]),
    }
    with model.trace(encoding):
        positions = nnsight.save(model.inputs[1]["position_ids"])
    torch.testing.assert_close(positions, torch.tensor([[0, 0, 1, 2]]))


@pytest.mark.parametrize("field", ["token_type_ids", "position_ids"])
@torch.no_grad()
def test_shared_sequence_field_broadcasts_across_rows(model, field):
    encoding = {
        "input_ids": torch.tensor([[2, 5, 4, 3], [2, 7, 4, 3]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        field: torch.tensor([[0, 1, 1, 1]])
        if field == "token_type_ids"
        else torch.tensor([[4, 5, 6, 7]]),
    }
    expected = model._module(**encoding)
    with model.trace(encoding):
        actual = nnsight.save(model.output.logits)
    torch.testing.assert_close(actual, expected.logits)
