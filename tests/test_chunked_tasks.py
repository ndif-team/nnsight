"""Chunked tasks on ``TransformersModel``.

Some tasks split one input into several encodings and forward each on its own:
token windows past the model's length limit in ``token-classification``, one
entailment pair per candidate label in ``zero-shot-classification``. Their
``preprocess`` yields those encodings rather than returning one, and they become
rows of the trace's single forward — so a read inside the block sees one row per
chunk, in the order the task yields them.

Two of these tasks take the pipeline's own input dict (``{"image": ...,
"question": ...}``) rather than model inputs, which is preprocessed here like any
other input; ``mask-generation``, whose preprocess runs the model to embed the
image before yielding one input per batch of points, has no single forward to
trace and is refused.
"""

import pytest
import torch

pytest.importorskip("PIL")

from PIL import Image

from nnsight.modeling.transformers import TransformersModel

NER = "hf-internal-testing/tiny-random-BertForTokenClassification"
NLI = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
OWLVIT = "hf-internal-testing/tiny-random-OwlViTForObjectDetection"
LAYOUTLM = "hf-internal-testing/tiny-random-LayoutLMForQuestionAnswering"
SAM = "hf-internal-testing/tiny-random-SamModel"

LABELS = ["travel", "cooking", "dancing"]


@pytest.fixture(scope="module")
def ner():
    return TransformersModel(NER, task="token-classification", dispatch=True)


@pytest.fixture(scope="module")
def nli():
    return TransformersModel(NLI, task="zero-shot-classification", dispatch=True)


class TestChunksAreRows:
    @torch.no_grad()
    def test_a_short_sentence_is_one_row(self, ner):
        with ner.trace("John lives in Paris"):
            logits = ner.output.logits.save()
        assert logits.shape[0] == 1

    @torch.no_grad()
    def test_a_long_sentence_is_one_row_per_window(self, ner):
        # `stride` makes the tokenizer return overflowing windows, and every
        # encoding the pipeline yields is a row of the forward. How many windows
        # a given max_length produces is the tokenizer's business and moves
        # across transformers versions, so the expected count comes from the
        # same preprocess call the trace makes.
        sentence = " ".join(["John lives in Paris"] * 200)
        pre_params, _, _ = ner.pipeline._sanitize_parameters(stride=16)
        windows = sum(1 for _ in ner.pipeline.preprocess(sentence, **pre_params))
        assert windows > 1  # the sentence really overflows into several windows
        with ner.trace(sentence, stride=16):
            logits = ner.output.logits.save()
        assert logits.shape[0] == windows

    @torch.no_grad()
    def test_one_row_per_candidate_label(self, nli):
        with nli.trace("one day I will see the world", candidate_labels=LABELS):
            logits = nli.output.logits.save()
        assert logits.shape[0] == len(LABELS)

    @torch.no_grad()
    def test_an_edit_reaches_every_chunk(self, nli):
        # The rows are the forward's, not a container the trace assembled after
        # it: an edit written once lands on all of them.
        with nli.trace("one day I will see the world", candidate_labels=LABELS):
            nli.output.logits = torch.zeros_like(nli.output.logits)
            logits = nli.output.logits.save()
        assert logits.shape[0] == len(LABELS)
        assert torch.all(logits == 0)

    @torch.no_grad()
    def test_several_prompts_in_one_invoke_still_batch(self, ner):
        with ner.trace(["John lives in Paris", "Mary works at Acme"]):
            logits = ner.output.logits.save()
        assert logits.shape[0] == 2


class TestChunkedBatching:
    @torch.no_grad()
    def test_a_chunked_invoke_cant_share_a_batch(self, nli):
        # The batcher gave this invoke one row before preprocessing decided on
        # three, so the second invoke's rows are not the ones it would read.
        with pytest.raises(NotImplementedError, match="splits this invoke into 3"):
            with nli.trace() as tracer:
                with tracer.invoke("one day I will see the world", candidate_labels=LABELS):
                    pass
                with tracer.invoke("I love to cook", candidate_labels=LABELS):
                    pass

    @torch.no_grad()
    def test_a_chunked_invoke_alone_is_the_whole_batch(self, nli):
        with nli.trace() as tracer:
            with tracer.invoke("one day I will see the world", candidate_labels=LABELS):
                logits = nli.output.logits.save()
        assert logits.shape[0] == len(LABELS)


class TestTaskInputDict:
    @torch.no_grad()
    def test_zero_shot_object_detection_takes_the_task_s_dict(self):
        model = TransformersModel(OWLVIT, task="zero-shot-object-detection", dispatch=True)
        with model.trace({"image": Image.new("RGB", (64, 64)), "candidate_labels": ["cat", "dog"]}):
            logits = model.output.logits.save()
        assert logits.shape[0] == 2  # one row per candidate label

    @torch.no_grad()
    def test_document_question_answering_takes_the_task_s_dict(self):
        model = TransformersModel(LAYOUTLM, task="document-question-answering", dispatch=True)
        document = {
            "image": None,
            "question": "What is it?",
            "word_boxes": [("hello", [0, 0, 10, 10]), ("world", [10, 0, 20, 10])],
        }
        with model.trace(document):
            start = model.output.start_logits.save()
        assert start.shape[0] == 1


class TestMaskGeneration:
    def test_tracing_an_image_is_refused(self):
        model = TransformersModel(SAM, task="mask-generation", dispatch=True)
        with pytest.raises(NotImplementedError, match="task='mask-generation'"):
            with model.trace(Image.new("RGB", (64, 64))):
                pass
