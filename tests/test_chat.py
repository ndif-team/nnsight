"""Chat-formatted input (a list of role/content messages) on ``TransformersModel``.

A chat is one conversation, not one row per message. Both ``generate`` and
``trace`` apply the model's chat template through the pipeline (``trace`` wraps
the messages in ``Chat`` the way ``Pipeline.__call__`` does), so a conversation
is tokenized correctly and counts as a single batch row.
"""


import pytest
import torch

from nnsight.modeling.transformers import TransformersModel

REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CHAT = [{"role": "user", "content": "Hello there"}]


@pytest.fixture(scope="module")
def chat_model():
    model = TransformersModel(REPO, task="text-generation", dispatch=True)
    assert model.tokenizer.chat_template is not None
    return model


class TestChat:
    def test_conversation_is_one_row(self, chat_model):
        # A chat is a single conversation, not one row per message.
        assert chat_model._batch_size(CHAT) == 1
        assert chat_model._batch_size([CHAT, CHAT]) == 2  # a batch of chats

    @torch.no_grad()
    def test_generate_from_chat(self, chat_model):
        with chat_model.pipe(CHAT, max_new_tokens=3, do_sample=False) as tracer:
            result = tracer.result.save()
        assert isinstance(result, list) and result

    @torch.no_grad()
    def test_trace_applies_chat_template(self, chat_model):
        # trace over a chat applies the template (via the pipeline) and reads logits.
        with chat_model.trace(CHAT):
            logits = chat_model.output.logits.save()
        assert logits.shape[0] == 1
        # The templated conversation is longer than the bare user text alone.
        with chat_model.trace("Hello there"):
            bare = chat_model.output.logits.save()
        assert logits.shape[1] != bare.shape[1]

    @torch.no_grad()
    def test_trace_list_of_conversations(self, chat_model):
        # A list of conversations is a batch of chats — one templated row per
        # conversation. Each conversation is a list of message dicts, i.e. a
        # list of lists, which `_is_pretokenized` would claim as pre-tokenized
        # sub-sequences if chat detection didn't run first (as in `_num_rows`).
        other = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "And of Germany?"},
        ]
        expected = [
            len(
                chat_model.tokenizer.apply_chat_template(
                    conv, add_generation_prompt=True, return_dict=False
                )
            )
            for conv in (CHAT, other)
        ]
        with chat_model.trace([CHAT, other]):
            inputs = chat_model.inputs.save()
            logits = chat_model.output.logits.save()
        assert logits.shape[0] == 2
        row_lengths = inputs[1]["attention_mask"].sum(dim=1).tolist()
        assert row_lengths == expected

    @torch.no_grad()
    def test_intervention_changes_chat_logits(self, chat_model):
        with chat_model.trace(CHAT):
            base = chat_model.output.logits[0, -1].save()
        with chat_model.trace(CHAT):
            chat_model.model.embed_tokens.output[:] = 0
            zeroed = chat_model.output.logits[0, -1].save()
        assert not torch.allclose(base, zeroed)


class TestPipelineOwnChatClass:
    """``any-to-any`` defines its own ``Chat`` and isinstance-checks it in
    preprocess; wrapping in the base ``Chat`` falls through to its raw-dict
    branch. The trace wraps with the pipeline module's own class instead."""

    REPO = "yujiepan/gemma-3n-tiny-random"

    @pytest.fixture(scope="class")
    def any_to_any(self):
        pytest.importorskip("PIL")
        pytest.importorskip("timm")  # the tiny checkpoint's vision tower
        # AnyToAnyPipeline requires librosa at construction whenever the model
        # has an audio input modality, even for a text/image-only call. librosa
        # needs numba, which trails new Python releases — hence a skip rather
        # than a dev dependency (CI installs it where numba exists).
        pytest.importorskip("librosa")
        return TransformersModel(self.REPO, task="any-to-any", dispatch=True)

    @torch.no_grad()
    def test_multimodal_chat_traces(self, any_to_any):
        from PIL import Image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (64, 64))},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        # The expected length comes from the same templating the pipeline does.
        encoded = any_to_any.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        with any_to_any.trace(messages):
            logits = any_to_any.output.logits.save()
        assert logits.shape[:2] == tuple(encoded["input_ids"].shape)

    @torch.no_grad()
    def test_plain_text_still_traces(self, any_to_any):
        with any_to_any.trace("Hello world"):
            logits = any_to_any.output.logits.save()
        assert logits.shape[0] == 1
