"""Vision-language models on a real ``TransformersModel`` (a tiny Llava).

A VLM uses the ``image-text-to-text`` pipeline, whose preprocessor is a
*processor* (image processor + tokenizer) rather than a bare tokenizer. This
exercises the processor-based loading path, tracing/generation over image+text
input, interventions on the vision tower / projector / language model, and the
input routing that keeps a multimodal encoding (with ``pixel_values``) from being
re-batched as if it were plain text.
"""


import pytest
import torch

pytest.importorskip("PIL")

from PIL import Image

from nnsight.modeling.transformers import TransformersModel

REPO = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
TASK = "image-text-to-text"
QUESTION = "Describe"


def _image():
    return Image.new("RGB", (64, 64), (120, 180, 60))


def _prompt(model):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": QUESTION}],
        }
    ]
    return model.processor.apply_chat_template(messages, add_generation_prompt=True)


def _encoding(model):
    """A full processor encoding (input_ids, attention_mask, pixel_values, ...)."""
    return model.processor(images=_image(), text=_prompt(model), return_tensors="pt")


@pytest.fixture(scope="module")
def llava():
    return TransformersModel(REPO, task=TASK, dispatch=True)


@pytest.fixture(scope="module")
def llava_meta():
    return TransformersModel(REPO, task=TASK)


class TestBuild:
    def test_processor_pipeline_loads(self, llava):
        # The image-text-to-text pipeline is processor-based; loading must source a
        # processor (not a stray tokenizer string, which the pipeline would reject).
        assert llava.processor is not None
        assert llava.tokenizer is not None  # derived from the processor
        assert llava.task == TASK

    def test_lazy_meta_build_tree(self, llava_meta):
        assert llava_meta.dispatched is False
        # The multimodal module tree is exposed as envoys.
        for name in ("vision_tower", "multi_modal_projector", "language_model"):
            assert hasattr(llava_meta.model, name)
        assert next(llava_meta.model.vision_tower.parameters()).device.type == "meta"

    def test_dispatch_loads_real_weights(self, llava):
        assert llava.dispatched is True
        assert next(llava.model.language_model.parameters()).device.type != "meta"


class TestTrace:
    @torch.no_grad()
    def test_reads_multimodal_activations(self, llava):
        enc = _encoding(llava)
        with llava.trace(**enc):
            projected = llava.model.multi_modal_projector.output.save()
            hidden = llava.model.language_model.layers[0].output[0].save()
            logits = llava.output.logits.save()
        # The projector maps image patches into the language embedding width.
        assert projected.shape[0] == 1
        assert hidden.ndim == 2
        assert logits.shape[0] == 1

    @torch.no_grad()
    def test_zeroing_projector_changes_logits(self, llava):
        enc = _encoding(llava)
        with llava.trace(**enc):
            base = llava.output.logits[0, -1].save()
        with llava.trace(**enc):
            llava.model.multi_modal_projector.output[:] = 0
            zeroed = llava.output.logits[0, -1].save()
        assert not torch.allclose(base, zeroed)

    @torch.no_grad()
    def test_unedited_trace_is_stable(self, llava):
        enc = _encoding(llava)
        with llava.trace(**enc):
            a = llava.output.logits.save()
        with llava.trace(**enc):
            b = llava.output.logits.save()
        assert torch.allclose(a, b)


class TestGenerate:
    @torch.no_grad()
    def test_generates_text_from_image(self, llava):
        kwargs = dict(
            text=_prompt(llava), images=_image(), max_new_tokens=3, do_sample=False
        )
        with llava.pipe(**kwargs) as tracer:
            result = tracer.result.save()
        assert isinstance(result[0]["generated_text"], str)

    @torch.no_grad()
    def test_intervention_changes_generation(self, llava):
        kwargs = dict(
            text=_prompt(llava), images=_image(), max_new_tokens=3, do_sample=False
        )
        with llava.pipe(**kwargs) as tracer:
            base = tracer.result.save()
        with llava.pipe(**kwargs) as tracer:
            llava.model.multi_modal_projector.output[:] = 0
            zeroed = tracer.result.save()
        assert base[0]["generated_text"] != zeroed[0]["generated_text"]


class TestScan:
    @torch.no_grad()
    def test_scan_reads_shapes_without_dispatch(self, llava_meta):
        enc = _encoding(llava_meta)
        with llava_meta.scan(**enc):
            projected = llava_meta.model.multi_modal_projector.output.save()
            logits = llava_meta.output.logits.save()
        assert "Fake" in type(logits).__name__
        assert logits.shape[0] == 1
        assert projected.shape[0] == 1
        assert llava_meta.dispatched is False


class TestInputRouting:
    """A multimodal encoding must pass straight to the model, not be re-tokenized
    and left-pad batched as if it were plain text (which corrupts image tokens)."""

    def test_multimodal_encoding_is_opaque(self, llava):
        enc = _encoding(llava)
        assert "pixel_values" in enc
        # A multimodal encoding is passed to the model as-is (every field), rather
        # than re-batched as text — which would drop pixel_values / corrupt ids.
        assert llava._is_opaque(None, dict(enc)) is True

    def test_text_only_encoding_is_pretokenized(self, llava):
        text_only = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        assert llava._is_pretokenized(None, text_only) is True

    def test_encoding_counts_one_row(self, llava):
        enc = _encoding(llava)
        assert llava._batch_size(**enc) == 1

    def test_keyword_image_text_counts_as_data(self, llava):
        # A VLM generate passes data by keyword; it must read as one data row, not
        # as params-only (which would wrongly expect invoke() blocks).
        assert llava._batch_size(text=_prompt(llava), images=_image()) == 1

    def test_params_only_still_zero_rows(self, llava):
        assert llava._batch_size(max_new_tokens=5) == 0

    def test_float_tensor_is_opaque(self, llava):
        # A float feature tensor passes straight through; integer token ids don't.
        assert llava._is_opaque(torch.randn(1, 3, 8, 8), {}) is True
        assert llava._is_pretokenized(torch.tensor([[1, 2, 3]]), {}) is True


@pytest.fixture(scope="module")
def vlm():
    from nnsight import VisionLanguageModel

    return VisionLanguageModel(REPO, dispatch=True)


class TestVisionLanguageModel:
    """The deprecated ``VisionLanguageModel`` still behaves as it did before the
    rewrite: like ``LanguageModel`` but with images, generating through the model
    and returning token ids. It runs the processor itself (the pipeline is not in
    the call), so a prompt plus images become the ids the model generated."""

    @torch.no_grad()
    def test_generate_returns_token_ids(self, vlm):
        with vlm.generate(
            text=_prompt(vlm), images=_image(), max_new_tokens=3, do_sample=False
        ) as tracer:
            out = tracer.result.save()
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 2 and out.shape[0] == 1

    @torch.no_grad()
    def test_prompt_positional_matches_text_kwarg(self, vlm):
        prompt = _prompt(vlm)
        with vlm.generate(prompt, images=_image(), max_new_tokens=3, do_sample=False) as t:
            positional = t.result.save()
        with vlm.generate(
            text=prompt, images=_image(), max_new_tokens=3, do_sample=False
        ) as t:
            keyword = t.result.save()
        assert torch.equal(positional, keyword)

    @torch.no_grad()
    def test_generation_is_greedy_by_default(self, vlm):
        kwargs = dict(text=_prompt(vlm), images=_image(), max_new_tokens=3)
        with vlm.generate(**kwargs) as t:
            first = t.result.save()
        with vlm.generate(**kwargs) as t:
            second = t.result.save()
        assert torch.equal(first, second)

    @torch.no_grad()
    def test_generator_output_is_the_result(self, vlm):
        with vlm.generate(
            text=_prompt(vlm), images=_image(), max_new_tokens=3, do_sample=False
        ) as tracer:
            through_generator = vlm.generator.output.save()
            returned = tracer.result.save()
        assert torch.equal(through_generator, returned)

    @torch.no_grad()
    def test_called_directly_returns_ids(self, vlm):
        out = vlm.generate(
            _prompt(vlm), images=_image(), max_new_tokens=3, do_sample=False
        )
        assert isinstance(out, torch.Tensor) and out.shape[0] == 1

    @torch.no_grad()
    def test_intervention_changes_generation(self, vlm):
        kwargs = dict(text=_prompt(vlm), images=_image(), max_new_tokens=3, do_sample=False)
        with vlm.generate(**kwargs) as tracer:
            clean = tracer.result.save()
        with vlm.generate(**kwargs) as tracer:
            vlm.model.multi_modal_projector.output[:] = 0
            zeroed = tracer.result.save()
        assert not torch.equal(clean, zeroed)

    def test_task_is_image_text_to_text(self, vlm):
        assert vlm.task == "image-text-to-text"


class TestProcessorInputs:
    """``trace``/``scan`` accept processor-style input (a prompt plus ``images=``),
    running the processor themselves the way ``generate`` always has.

    Before this, such a call was forwarded to the model untouched and raised from
    inside modeling code (``You must specify exactly one of input_ids or
    inputs_embeds``), which pointed nowhere near the real problem.
    """

    def test_trace_with_positional_prompt_and_images(self, llava):
        with llava.trace(_prompt(llava), images=[_image()]):
            logits = llava.output.logits.save()

        assert logits.shape[0] == 1
        assert logits.shape[-1] == llava.config.text_config.vocab_size

    def test_trace_with_text_keyword_and_images(self, llava):
        with llava.trace(text=_prompt(llava), images=[_image()]):
            logits = llava.output.logits.save()

        assert logits.shape[0] == 1

    def test_processor_input_matches_a_prebuilt_encoding(self, llava):
        """The convenience form must produce exactly what the manual form does."""
        with llava.trace(_prompt(llava), images=[_image()]):
            from_processor = llava.output.logits.save()

        with llava.trace(_encoding(llava)):
            from_encoding = llava.output.logits.save()

        assert torch.allclose(from_processor, from_encoding)

    def test_interventions_apply_to_processor_input(self, llava):
        with llava.trace(_prompt(llava), images=[_image()]):
            projected = llava.model.multi_modal_projector.output.save()

        assert projected.ndim == 3

    def test_scan_with_images(self, llava):
        """Featurizing an image needs numpy, which a fake-tensor mode refuses — the
        processor has to run outside the scan's `FakeTensorMode`."""
        with llava.scan(_prompt(llava), images=[_image()]):
            logits = llava.output.logits.shape.save()

        assert logits[0] == 1

    def test_prompt_given_twice_is_rejected(self, llava):
        with pytest.raises(ValueError, match="both positionally and as `text=`"):
            with llava.trace(_prompt(llava), text=_prompt(llava), images=[_image()]):
                pass

    def test_text_only_trace_is_unaffected(self, llava):
        """No media keyword means the ordinary tokenization path, not the processor."""
        with llava.trace("just text, no image"):
            logits = llava.output.logits.save()

        assert logits.shape[0] == 1
