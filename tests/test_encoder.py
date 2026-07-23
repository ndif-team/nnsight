"""Encoder tasks (fill-mask, text-classification) on ``TransformersModel``.

These are non-causal: the tokenizer keeps its default right padding and no
left-pad ``position_ids`` correction is applied (unlike a causal decoder). This
exercises the task-aware padding and the pipeline-preprocess forward path for
encoder models, plus the pipeline's own input validation being inherited.
"""


import pytest
import torch

from nnsight.modeling.transformers import TransformersModel

MASKED_LM = "hf-internal-testing/tiny-random-BertForMaskedLM"
CLASSIFIER = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"


@pytest.fixture(scope="module")
def bert():
    return TransformersModel(MASKED_LM, task="fill-mask", dispatch=True)


@pytest.fixture(scope="module")
def classifier():
    return TransformersModel(CLASSIFIER, task="text-classification", dispatch=True)


@pytest.fixture(scope="module")
def embedder():
    # feature-extraction drops the task head and runs the base encoder, so the
    # output is the raw hidden states rather than logits.
    return TransformersModel(MASKED_LM, task="feature-extraction", dispatch=True)


class TestEncoderPadding:
    def test_encoder_is_not_causal_and_right_pads(self, classifier):
        assert classifier._is_causal() is False
        assert classifier.tokenizer.padding_side == "right"

    @torch.no_grad()
    def test_batched_classification(self, classifier):
        with classifier.trace() as tracer:
            with tracer.invoke("great"):
                a = classifier.output.logits.save()
            with tracer.invoke("a much longer and more detailed review of the film"):
                b = classifier.output.logits.save()
        assert a.shape[0] == 1 and b.shape[0] == 1
        assert a.shape[1] == b.shape[1]  # same class count

    @torch.no_grad()
    def test_batched_matches_solo(self, classifier):
        # A right-padded batched row matches the prompt run alone: the attention
        # mask hides the padding, so no position_ids correction is needed.
        short, long = "good", "an extended and elaborate description here now"
        with classifier.trace(short):
            solo = classifier.output.logits.save()
        with classifier.trace() as tracer:
            with tracer.invoke(short):
                batched = classifier.output.logits.save()
            with tracer.invoke(long):
                pass
        assert torch.allclose(batched, solo, atol=1e-4)

    @torch.no_grad()
    def test_intervention_changes_logits(self, classifier):
        with classifier.trace("a decent film"):
            base = classifier.output.logits.save()
        with classifier.trace("a decent film"):
            classifier.distilbert.embeddings.output[:] = 0
            zeroed = classifier.output.logits.save()
        assert not torch.allclose(base, zeroed)


class TestFillMask:
    @torch.no_grad()
    def test_reads_logits(self, bert):
        mask = bert.tokenizer.mask_token
        with bert.trace(f"Paris is the {mask} of France."):
            logits = bert.output.logits.save()
        assert logits.shape[0] == 1

    @torch.no_grad()
    def test_batched_fill_mask_right_padded(self, bert):
        mask = bert.tokenizer.mask_token
        with bert.trace() as tracer:
            with tracer.invoke(f"The {mask} sat."):
                a = bert.output.logits.save()
            with tracer.invoke(f"A considerably longer sentence with a {mask} in it."):
                b = bert.output.logits.save()
        assert a.shape[1] == b.shape[1]  # common right-padded sequence length

    def test_missing_mask_raises(self, bert):
        # The fill-mask pipeline requires a [MASK]; that validation is inherited
        # because we preprocess through the pipeline.
        with pytest.raises(Exception, match="mask"):
            with bert.trace("no mask token here"):
                pass


class TestPipe:
    """``pipe`` runs the whole task pipeline, so an encoder returns the task's
    postprocessed records — labels for a classifier, ranked fills for fill-mask —
    where ``trace`` (and generate, which these tasks don't have) returns raw
    tensors."""

    @torch.no_grad()
    def test_classifier_pipe_returns_label_records(self, classifier):
        with classifier.pipe("a decent film") as tracer:
            result = tracer.result.save()
        assert isinstance(result, list)
        assert "label" in result[0] and "score" in result[0]

    @torch.no_grad()
    def test_fill_mask_pipe_returns_ranked_fills(self, bert):
        mask = bert.tokenizer.mask_token
        with bert.pipe(f"Paris is the {mask} of France.") as tracer:
            result = tracer.result.save()
        assert isinstance(result, list) and result
        assert {"score", "token", "token_str", "sequence"} <= set(result[0])

    @torch.no_grad()
    def test_pipe_is_traced(self, classifier):
        # pipe runs the real forward, so an intervention in the block still lands —
        # zeroing the embeddings shifts the pipeline's own score.
        with classifier.pipe("a decent film") as tracer:
            base = tracer.result.save()
        with classifier.pipe("a decent film") as tracer:
            classifier.distilbert.embeddings.output[:] = 0
            zeroed = tracer.result.save()
        assert base[0]["score"] != zeroed[0]["score"]


class TestFeatureExtraction:
    """feature-extraction runs the base encoder (no task head): the output is the
    encoder's hidden states, and interventions on a layer still apply."""

    @torch.no_grad()
    def test_output_is_hidden_states(self, embedder):
        with embedder.trace("hello world"):
            out = embedder.output.save()
        # A ModelOutput carrying last_hidden_state, not logits.
        assert hasattr(out, "last_hidden_state")
        assert out.last_hidden_state.ndim == 3

    @torch.no_grad()
    def test_batched_embeddings(self, embedder):
        with embedder.trace() as tracer:
            with tracer.invoke("great"):
                a = embedder.output.last_hidden_state.save()
            with tracer.invoke("a longer piece of input text"):
                b = embedder.output.last_hidden_state.save()
        assert a.shape[0] == 1 and b.shape[0] == 1
