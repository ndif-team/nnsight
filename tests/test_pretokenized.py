"""Pretokenized traces preserve the model inputs produced by a tokenizer."""

import copy

import numpy as np
import pytest
import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
)

import nnsight
from nnsight.modeling import processing
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


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(torch.tensor([[2, 5, 6, 7, 5, 3], [2, 6, 3, -100, -100, -100]]), id="wider-than-the-ids"),
        pytest.param(torch.tensor([[2], [0]]), id="one-column"),
    ],
)
@torch.no_grad()
def test_labels_that_are_not_per_token_keep_their_own_width(model, labels):
    # Labels are padded to the ids' width only when they label the ids' tokens;
    # a target with its own length (seq2seq, one column per example) is not
    # cut or padded to match.
    ids = torch.tensor([[2, 5, 4, 3], [2, 7, 4, 3]])
    with model.trace() as tracer:
        with tracer.invoke(ids[:1], labels=labels[:1]):
            pass
        with tracer.invoke(ids[1:], labels=labels[1:]):
            pass
        with tracer.invoke():
            received = nnsight.save(model.inputs[1]["labels"])
            tracer.stop()
    torch.testing.assert_close(received, labels)


# -- one rule for every input form and every per-row field -------------------


@pytest.fixture
def causal(tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(7)
    module = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_layer=1, n_head=2, n_embd=16)).eval()
    return TransformersModel(module, task="text-generation", tokenizer=tokenizer, device="cpu")


@pytest.fixture
def seq2seq(tokenizer):
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(7)
    config = T5Config(
        vocab_size=8, d_model=16, d_kv=4, d_ff=32, num_layers=1, num_heads=2,
        decoder_start_token_id=0, pad_token_id=0,
    )
    module = T5ForConditionalGeneration(config).eval()
    # Its own tokenizer: a model sets the padding side on the one it is given.
    return TransformersModel(
        module, task="text-generation", tokenizer=copy.deepcopy(tokenizer), device="cpu"
    )


def batch(model, *invokes):
    """Trace each (args, kwargs) as its own invoke -> (inputs received, output)."""
    with model.trace() as tracer:
        for args, kwargs in invokes:
            with tracer.invoke(*args, **kwargs):
                pass
        with tracer.invoke():
            received = nnsight.save(model.inputs[1])
            output = nnsight.save(model.output)
    return received, output


@pytest.mark.parametrize("as_tensor", [False, True], ids=["lists", "tensors"])
@torch.no_grad()
def test_unbatched_ids_take_unbatched_fields(causal, as_tensor):
    # Beside flat ids, flat labels are per-token labels for that one row.
    ids = [2, 5, 6, 7, 3]
    value = torch.tensor(ids) if as_tensor else ids
    expected = causal._module(input_ids=torch.tensor([ids]), labels=torch.tensor([ids]))
    with causal.trace(value, labels=value):
        loss = nnsight.save(causal.output.loss)
    torch.testing.assert_close(loss, expected.loss)


@pytest.mark.parametrize("label", [1, torch.tensor(1), torch.tensor([1])], ids=["int", "0-d", "1-d"])
@torch.no_grad()
def test_one_class_label_beside_unbatched_ids(tokenizer, label):
    module = BertForSequenceClassification(tiny_config()).eval()
    model = TransformersModel(module, task="text-classification", tokenizer=tokenizer, device="cpu")
    expected = module(input_ids=torch.tensor([[2, 5, 6, 3]]), labels=torch.tensor([1]))
    with model.trace([2, 5, 6, 3], labels=label):
        loss = nnsight.save(model.output.loss)
    torch.testing.assert_close(loss, expected.loss)


@torch.no_grad()
def test_an_invoke_without_a_field_gets_its_default(model):
    # A tokenizer's encoding carries token_type_ids; plain ids beside it do not.
    encoding = model.tokenizer("hello [MASK] world", return_tensors="pt")
    ids = torch.tensor([[2, 5, 4, 3]])
    received, output = batch(model, ((encoding,), {}), ((ids,), {}), (("hello [MASK]",), {}))
    assert received["input_ids"].shape == (3, 5)
    assert not received["token_type_ids"].any()
    alone = model._module(input_ids=ids).logits
    torch.testing.assert_close(output.logits[1, :4], alone[0], atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_an_unlabelled_invoke_is_ignored_by_the_loss(causal):
    labelled = torch.tensor([[2, 5, 3]])
    received, output = batch(
        causal, ((labelled,), {"labels": labelled}), ((torch.tensor([[2, 5, 6, 7, 3]]),), {})
    )
    assert received["labels"].tolist() == [[-100, -100, 2, 5, 3], [-100] * 5]
    expected = causal._module(**received)
    torch.testing.assert_close(output.loss, expected.loss)


@torch.no_grad()
def test_a_field_with_no_default_must_be_in_every_invoke(seq2seq):
    ids = torch.tensor([[2, 5, 3]])
    with pytest.raises(ValueError, match="1 of 2 rows have `decoder_input_ids`"):
        batch(seq2seq, ((ids,), {"decoder_input_ids": torch.tensor([[0, 5]])}), ((ids,), {}))


@pytest.mark.parametrize("form", ["encoding", "keywords"])
@torch.no_grad()
def test_decoder_fields_are_batched_at_their_own_width(seq2seq, form):
    short = {
        "input_ids": torch.tensor([[2, 5, 6, 3]]),
        "decoder_input_ids": torch.tensor([[0, 5, 6]]),
        "decoder_attention_mask": torch.tensor([[1, 1, 1]]),
    }
    long = {
        "input_ids": torch.tensor([[2, 7, 3]]),
        "decoder_input_ids": torch.tensor([[0, 5, 6, 7, 5]]),
        "decoder_attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    invokes = [((row,), {}) if form == "encoding" else ((), row) for row in (short, long)]
    received, output = batch(seq2seq, *invokes)
    # Padded on the right, whichever side the input is padded on.
    assert received["decoder_input_ids"].tolist() == [[0, 5, 6, 0, 0], [0, 5, 6, 7, 5]]
    assert received["decoder_attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1] * 5]
    alone = seq2seq._module(**short).logits
    torch.testing.assert_close(output.logits[0, :3], alone[0], atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_seq2seq_labels_are_padded_on_the_right(seq2seq):
    received, _ = batch(
        seq2seq,
        ((torch.tensor([[2, 5, 6, 3]]),), {"labels": torch.tensor([[5, 3]])}),
        ((torch.tensor([[2, 7, 3]]),), {"labels": torch.tensor([[5, 6, 7, 3]])}),
    )
    assert received["labels"].tolist() == [[5, 3, -100, -100], [5, 6, 7, 3]]


@torch.no_grad()
def test_per_example_fields_outside_the_text_ones_are_batched(tokenizer):
    from transformers import BertForQuestionAnswering

    module = BertForQuestionAnswering(tiny_config()).eval()
    model = TransformersModel(module, task="fill-mask", tokenizer=tokenizer, device="cpu")
    ids = torch.tensor([[2, 5, 6, 3], [2, 7, 3, 0]])
    expected = module(
        input_ids=ids, start_positions=torch.tensor([1, 0]), end_positions=torch.tensor([2, 1])
    )
    _, output = batch(
        model,
        ((ids[:1],), {"start_positions": torch.tensor([1]), "end_positions": torch.tensor([2])}),
        ((ids[1:],), {"start_positions": torch.tensor([0]), "end_positions": torch.tensor([1])}),
    )
    torch.testing.assert_close(output.loss, expected.loss)


@torch.no_grad()
def test_inputs_embeds_trace_and_batch_like_ids(causal):
    ids = torch.tensor([[2, 5, 6], [2, 7, 3]])
    embeds = causal._module.get_input_embeddings()(ids)
    expected = causal._module(inputs_embeds=embeds).logits
    with causal.trace(inputs_embeds=embeds):
        logits = nnsight.save(causal.output.logits)
    torch.testing.assert_close(logits, expected)

    # Two invokes of different lengths: left-padded, positions counted from each
    # row's first token, so a row reads as it does alone.
    received, output = batch(
        causal, ((), {"inputs_embeds": embeds[0, :2]}), ((), {"inputs_embeds": embeds[1:]})
    )
    assert received["inputs_embeds"].shape == (2, 3, 16)
    assert received["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert received["position_ids"].tolist() == [[0, 0, 1], [0, 1, 2]]
    alone = causal._module(inputs_embeds=embeds[:1, :2]).logits
    torch.testing.assert_close(output.logits[0, -2:], alone[0], atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_generate_from_inputs_embeds(causal):
    embeds = causal._module.get_input_embeddings()(torch.tensor([[2, 5, 6]]))
    expected = causal._module.generate(
        inputs_embeds=embeds, max_new_tokens=3, do_sample=False, pad_token_id=0
    )
    with causal.generate(inputs_embeds=embeds, max_new_tokens=3, do_sample=False) as tracer:
        result = nnsight.save(tracer.result)
    assert result.tolist() == expected.tolist()


@torch.no_grad()
def test_positions_are_derived_only_from_left_padding(causal):
    ids = torch.tensor([[2, 5, 6, 7, 3]])
    for mask, expected in (
        ([[0, 0, 1, 1, 1]], [[0, 0, 0, 1, 2]]),  # padding then tokens
        ([[1, 1, 0, 1, 1]], None),  # a gap is not padding
        ([[1, 1, 1, 0, 0]], None),  # right padding is already correct
    ):
        with causal.trace(ids, attention_mask=torch.tensor(mask)):
            received = nnsight.save(causal.inputs[1])
        positions = received.get("position_ids")
        assert (positions.tolist() if positions is not None else None) == expected


@torch.no_grad()
def test_a_flag_beside_keyword_ids_changes_nothing_else(causal):
    ids, mask = torch.tensor([[0, 2, 5]]), torch.tensor([[0, 1, 1]])
    with causal.trace(input_ids=ids, attention_mask=mask, output_hidden_states=True):
        received = nnsight.save(causal.inputs[1])
    assert received["position_ids"].tolist() == [[0, 0, 1]]
    assert received["output_hidden_states"] is True


@torch.no_grad()
def test_equal_length_rows_batch_without_a_pad_token(causal):
    causal.tokenizer.pad_token = None
    received, _ = batch(causal, ((torch.tensor([[2, 5]]),), {}), ((torch.tensor([[6, 3]]),), {}))
    assert received["input_ids"].tolist() == [[2, 5], [6, 3]]
    with pytest.raises(ValueError, match="without a pad token"):
        batch(causal, ((torch.tensor([[2, 5]]),), {}), ((torch.tensor([[6]]),), {}))


# -- the edges an audit against the native models turned up -------------------


@torch.no_grad()
def test_one_class_label_is_one_shape_however_it_is_written(tokenizer):
    module = BertForSequenceClassification(tiny_config()).eval()
    model = TransformersModel(module, task="text-classification", tokenizer=tokenizer, device="cpu")
    received, _ = batch(
        model,
        (([2, 5, 6, 3],), {"labels": 1}),
        (([2, 7, 3],), {"labels": torch.tensor(0)}),
        (([2, 6, 3],), {"labels": [1]}),
        ((torch.tensor([[2, 5, 3]]),), {"labels": torch.tensor([0])}),
    )
    assert received["labels"].tolist() == [1, 0, 1, 0]


@torch.no_grad()
def test_only_segments_and_positions_are_shared_across_rows(tokenizer):
    module = BertForSequenceClassification(tiny_config()).eval()
    model = TransformersModel(module, task="text-classification", tokenizer=tokenizer, device="cpu")
    ids = torch.tensor([[2, 5, 6, 3], [2, 7, 6, 3]])
    # The model raises for one label over two rows; so does the trace.
    with pytest.raises(ValueError, match="`labels` has 1 rows, but `input_ids` has 2"):
        with model.trace(ids, labels=torch.tensor([1])):
            pass
    with pytest.raises(ValueError, match="`token_type_ids` has 3 positions for a row of 4"):
        with model.trace(ids, token_type_ids=torch.tensor([[0, 1, 1]])):
            pass


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([torch.tensor([[2, 5, 4, 3]]), torch.tensor([[2, 7, 3]])], id="list-of-1xL-tensors"),
        pytest.param(np.array([[2, 5, 4, 3], [2, 7, 3, 0]]), id="numpy"),
        pytest.param([torch.tensor([[2, 5, 3], [2, 6, 3]]), torch.tensor([[2, 7, 3]])], id="list-of-batches"),
        pytest.param({"input_ids": [[2, 5, 4, 3], [2, 7, 3]]}, id="encoding-of-lists"),
        pytest.param([np.int64(2), np.int64(5), np.int64(3)], id="numpy-ints"),
    ],
)
@torch.no_grad()
def test_id_forms_count_and_split_into_the_same_rows(model, ids):
    rows = model._batch_size(ids)
    received, _ = batch(model, ((ids,), {}), ((torch.tensor([[2, 5, 3]]),), {}))
    assert received["input_ids"].dim() == 2
    assert received["input_ids"].shape[0] == rows + 1


@torch.no_grad()
def test_a_tensor_outside_the_table_passes_the_input_through_in_every_form(causal):
    # Not known to be per-row, so the input is handed over whole — whichever
    # way it was written — and such invokes do not batch.
    ids, extra = torch.tensor([[2, 5, 6]]), torch.arange(3)
    for args, kwargs in (((ids,), {}), ((), {"input_ids": ids}), (({"input_ids": ids},), {})):
        data = args[0] if args else None
        rows, _ = processing.preprocess_invoke(causal, data, {**kwargs, "cache_position": extra})
        assert rows is None
    # Ids written as lists still reach the model as the tensors it takes.
    expected = causal._module(input_ids=ids, cache_position=extra).logits
    for args, kwargs in ((([2, 5, 6],), {}), (({"input_ids": [[2, 5, 6]]},), {})):
        with causal.trace(*args, **kwargs, cache_position=extra):
            logits = nnsight.save(causal.output.logits)
        torch.testing.assert_close(logits, expected)


@torch.no_grad()
def test_a_tokenizer_encoding_without_tensors_is_an_encoding(model):
    encoding = model.tokenizer("hello [MASK] world")
    assert isinstance(encoding["input_ids"], list)
    expected = model._module(**model.tokenizer("hello [MASK] world", return_tensors="pt")).logits
    with model.trace(encoding):
        logits = nnsight.save(model.output.logits)
    torch.testing.assert_close(logits, expected)


@torch.no_grad()
def test_no_mask_is_assumed_beside_a_cache(causal):
    module = causal._module
    past = module(input_ids=torch.tensor([[2, 5, 6]]), use_cache=True).past_key_values
    expected = module(input_ids=torch.tensor([[7]]), past_key_values=copy.deepcopy(past)).logits
    with causal.trace([7], past_key_values=past):
        received = nnsight.save(causal.inputs[1])
        logits = nnsight.save(causal.output.logits)
    assert "attention_mask" not in received
    torch.testing.assert_close(logits, expected)


@torch.no_grad()
def test_a_mask_wider_than_the_input_gets_no_positions(causal):
    # A key-value cache continuation: the mask covers past and new tokens.
    module = causal._module
    past = module(input_ids=torch.tensor([[0, 2, 5]]), attention_mask=torch.tensor([[0, 1, 1]]), use_cache=True)
    new, mask = torch.tensor([[6]]), torch.tensor([[0, 1, 1, 1]])
    # The forward appends to the cache it is given, so each run gets its own.
    cache = copy.deepcopy(past.past_key_values)
    expected = module(input_ids=new, attention_mask=mask, past_key_values=cache).logits
    with causal.trace(new, attention_mask=mask, past_key_values=past.past_key_values):
        received = nnsight.save(causal.inputs[1])
        logits = nnsight.save(causal.output.logits)
    assert "position_ids" not in received
    torch.testing.assert_close(logits, expected)


@pytest.mark.parametrize("dtype", [torch.long, torch.bool])
@torch.no_grad()
def test_positions_are_decided_row_by_row(causal, dtype):
    # A row padded on the right, a row with a gap, and a row nnsight left-pads:
    # only the last is counted from its first token.
    received, _ = batch(
        causal,
        ((torch.tensor([[2, 5, 6, 0, 0]]),), {"attention_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=dtype)}),
        ((torch.tensor([[2, 5, 6, 7, 3]]),), {"attention_mask": torch.tensor([[1, 1, 0, 1, 1]], dtype=dtype)}),
        ((torch.tensor([[2, 7, 3]]),), {"attention_mask": torch.tensor([[1, 1, 1]], dtype=dtype)}),
    )
    assert received["position_ids"].tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 0, 0, 1, 2]]


@torch.no_grad()
def test_default_positions_count_from_a_rows_first_token(causal):
    # One invoke passes positions, so the other's are filled in — from its mask.
    received, _ = batch(
        causal,
        ((torch.tensor([[0, 2, 5]]),), {"attention_mask": torch.tensor([[0, 1, 1]])}),
        ((torch.tensor([[2, 5, 6]]),), {"position_ids": torch.tensor([[4, 5, 6]])}),
    )
    assert received["position_ids"].tolist() == [[0, 0, 1], [4, 5, 6]]


@torch.no_grad()
def test_each_field_is_padded_with_its_own_value(causal, seq2seq):
    embeds = causal._module.get_input_embeddings()(torch.tensor([[2, 5, 6]])) + 1.0
    received, _ = batch(
        causal,
        ((), {"inputs_embeds": embeds[:, :1], "special_tokens_mask": [[0]], "global_attention_mask": [[1]]}),
        ((), {"inputs_embeds": embeds, "special_tokens_mask": [[0, 0, 0]], "global_attention_mask": [[1, 1, 1]]}),
    )
    assert not received["inputs_embeds"][0, :2].any()
    assert received["special_tokens_mask"].tolist() == [[1, 1, 0], [0, 0, 0]]
    assert received["global_attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1]]

    decoder = seq2seq._module.get_input_embeddings()(torch.tensor([[0, 5, 6]])) + 1.0
    ids = torch.tensor([[2, 5, 3]])
    received, _ = batch(
        seq2seq,
        ((ids,), {"decoder_inputs_embeds": decoder[:, :1]}),
        ((ids,), {"decoder_inputs_embeds": decoder}),
    )
    assert received["decoder_inputs_embeds"].shape == (2, 3, 16)
    assert not received["decoder_inputs_embeds"][0, 1:].any()


@torch.no_grad()
def test_ids_of_more_than_two_dimensions_get_a_mask_per_token(tokenizer):
    from transformers import BertForMultipleChoice

    module = BertForMultipleChoice(tiny_config()).eval()
    model = TransformersModel(module, task="fill-mask", tokenizer=tokenizer, device="cpu")
    ids = torch.tensor([[[2, 5, 6, 3], [2, 7, 6, 3]]])
    with model.trace(ids):
        received = nnsight.save(model.inputs[1])
    assert received["attention_mask"].shape == ids.shape
