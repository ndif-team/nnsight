"""Chunked tasks on ``TransformersModel``.

Some tasks split one input into several encodings and forward each on its own:
token windows past the model's length limit in ``token-classification``, one
entailment pair per candidate label in ``zero-shot-classification``. Their
``preprocess`` yields those encodings rather than returning one, and they become
rows of the trace's single forward — so a read inside the block sees one row per
chunk, in the order the task yields them.

Some tasks take the pipeline's own input dict (``{"image": ..., "question":
...}``) rather than model inputs, which is preprocessed here like any other
input — through the pipeline's ``_args_parser`` first when it has one, since
that is where ``table-question-answering`` builds its ``pd.DataFrame``. A
dual-encoder zero-shot task (CLIP, CLAP) nests the candidate labels' text
encoding inside its preprocess row, and those tensors are merged into the
forward rather than silently dropped.

Two tasks are refused: ``mask-generation``, whose preprocess runs the model to
embed the image before yielding one input per batch of points, has no single
forward to trace; ``keypoint-matching`` takes a pair of images as one input,
which the list convention (one prompt per element) would split.
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
TAPAS = "hf-internal-testing/tiny-random-TapasForQuestionAnswering"
CLIP = "hf-internal-testing/tiny-random-CLIPModel"
CLAP = "hf-internal-testing/tiny-clap-htsat-unfused"
SUPERGLUE = "magic-leap-community/superglue_outdoor"

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

    @torch.no_grad()
    def test_table_question_answering_takes_the_task_s_dict(self):
        # The dict->DataFrame step lives in the pipeline's _args_parser (run
        # by __call__, not preprocess), so the trace has to route the task
        # dict through it. The expected width comes from the same parse +
        # preprocess the trace makes.
        model = TransformersModel(TAPAS, task="table-question-answering", dispatch=True)
        task_input = {
            "table": {
                "City": ["Paris", "London"],
                "Population": ["2000000", "9000000"],
            },
            "query": "Which city has the biggest population?",
        }
        parsed = model.pipeline._args_parser(dict(task_input))[0]
        encoded = model.pipeline.preprocess(parsed)
        with model.trace(task_input):
            logits = model.output.logits.save()
        assert logits.shape == encoded["input_ids"].shape


class TestDualEncoderZeroShot:
    # These pipelines nest the candidate labels' text encoding inside their
    # preprocess row and unwrap it in _forward; the trace merges those tensors
    # into its one forward, giving one text row per candidate label against
    # the single image/audio row.

    @torch.no_grad()
    def test_zero_shot_image_classification_keeps_the_text_half(self):
        model = TransformersModel(CLIP, task="zero-shot-image-classification", dispatch=True)
        labels = ["cat", "dog", "bird"]
        with model.trace(Image.new("RGB", (64, 64)), candidate_labels=labels):
            per_image = model.output.logits_per_image.save()
        assert per_image.shape == (1, len(labels))

    @torch.no_grad()
    def test_zero_shot_audio_classification_keeps_the_text_half(self):
        import numpy as np

        model = TransformersModel(CLAP, task="zero-shot-audio-classification", dispatch=True)
        labels = ["speech", "music"]
        audio = np.zeros(16000, dtype=np.float32)
        with model.trace(audio, candidate_labels=labels):
            per_audio = model.output.logits_per_audio.save()
        assert per_audio.shape == (1, len(labels))


class TestMaskGeneration:
    def test_tracing_an_image_is_refused(self):
        model = TransformersModel(SAM, task="mask-generation", dispatch=True)
        with pytest.raises(NotImplementedError, match="task='mask-generation'"):
            with model.trace(Image.new("RGB", (64, 64))):
                pass


class TestKeypointMatching:
    MESSAGE = (
        "task='keypoint-matching' takes a pair of images as one input, which "
        "a trace's list convention (one prompt per element) would split. Run "
        "the whole task with model.pipe([image_a, image_b]), or trace one "
        "forward on an encoding you build yourself: "
        "model.image_processor(images=[image_a, image_b], "
        "return_tensors='pt')."
    )

    @pytest.fixture(scope="class")
    def matcher(self):
        return TransformersModel(SUPERGLUE, task="keypoint-matching", dispatch=True)

    def test_tracing_a_pair_of_images_is_refused(self, matcher):
        pair = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with pytest.raises(NotImplementedError) as excinfo:
            with matcher.trace(pair):
                pass
        assert str(excinfo.value) == self.MESSAGE

    def test_tracing_a_nested_pair_is_refused_too(self, matcher):
        # A nested pair would otherwise be read as pre-tokenized ids.
        pair = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with pytest.raises(NotImplementedError) as excinfo:
            with matcher.trace([pair]):
                pass
        assert str(excinfo.value) == self.MESSAGE
