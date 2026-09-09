"""Image-classification (a vision-only pipeline) on ``TransformersModel``.

The preprocessor here is an image processor (no tokenizer). Raw PIL images are
turned into ``pixel_values`` by the pipeline's ``preprocess``, and several images
are batched by the pipeline's ``pad_collate_fn`` (concatenated on dim 0). This
exercises the media preprocess path and image batching.
"""


import pytest
import torch

pytest.importorskip("PIL")

from PIL import Image

from nnsight.modeling.transformers import TransformersModel

REPO = "hf-internal-testing/tiny-random-ViTForImageClassification"


def _image(color):
    return Image.new("RGB", (64, 64), color)


@pytest.fixture(scope="module")
def vit():
    return TransformersModel(REPO, task="image-classification", dispatch=True)


class TestImageClassification:
    def test_vision_pipeline_has_no_tokenizer(self, vit):
        assert vit.tokenizer is None
        assert vit.image_processor is not None
        assert vit._is_causal() is False

    @torch.no_grad()
    def test_single_image(self, vit):
        with vit.trace(_image((10, 120, 200))):
            logits = vit.output.logits.save()
        assert logits.shape[0] == 1

    @torch.no_grad()
    def test_batched_images(self, vit):
        # Two image invokes combine into one batch; the empty invoke sees both rows.
        with vit.trace() as tracer:
            with tracer.invoke(_image((10, 120, 200))):
                a = vit.output.logits.save()
            with tracer.invoke(_image((200, 10, 10))):
                pass
            with tracer.invoke():
                whole = vit.output.logits.save()
        assert a.shape[0] == 1
        assert whole.shape[0] == 2  # both images' rows

    @torch.no_grad()
    def test_reads_vision_activations(self, vit):
        with vit.trace(_image((10, 120, 200))):
            embedded = vit.vit.embeddings.output.save()
        assert embedded.shape[0] == 1  # one image's patch embeddings

    @torch.no_grad()
    def test_intervention_changes_logits(self, vit):
        image = _image((10, 120, 200))
        with vit.trace(image):
            base = vit.output.logits.save()
        with vit.trace(image):
            vit.vit.embeddings.output[:] = 0
            zeroed = vit.output.logits.save()
        assert not torch.allclose(base, zeroed)


class TestPipe:
    """``pipe`` runs the whole pipeline, so image-classification returns the task's
    postprocessed label records (there is no generate for this task)."""

    @torch.no_grad()
    def test_single_image_pipe_returns_label_records(self, vit):
        with vit.pipe(_image((10, 120, 200))) as tracer:
            result = tracer.result.save()
        assert isinstance(result, list)
        assert "label" in result[0] and "score" in result[0]

    @torch.no_grad()
    def test_batched_images_pipe_one_record_per_image(self, vit):
        # A list of images is one pipe call; the pipeline returns a record set per
        # image.
        with vit.pipe([_image((10, 120, 200)), _image((200, 10, 10))]) as tracer:
            result = tracer.result.save()
        assert len(result) == 2
        assert all("label" in per_image[0] for per_image in result)

    @torch.no_grad()
    def test_pipe_is_traced(self, vit):
        # pipe runs the real forward, so an intervention in the block still lands.
        image = _image((10, 120, 200))
        with vit.pipe(image) as tracer:
            base = tracer.result.save()
        with vit.pipe(image) as tracer:
            vit.vit.embeddings.output[:] = 0
            zeroed = tracer.result.save()
        assert base[0]["score"] != zeroed[0]["score"]
