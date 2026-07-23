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
    def test_intervention_changes_chat_logits(self, chat_model):
        with chat_model.trace(CHAT):
            base = chat_model.output.logits[0, -1].save()
        with chat_model.trace(CHAT):
            chat_model.model.embed_tokens.output[:] = 0
            zeroed = chat_model.output.logits[0, -1].save()
        assert not torch.allclose(base, zeroed)
