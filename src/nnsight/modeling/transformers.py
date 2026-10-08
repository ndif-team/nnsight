"""HuggingFace models, whatever the task, without knowing the task.

Reading a model means getting an input into it. A prompt has to be tokenized, an
image featurized, chat messages templated; a batch of them has to be padded to a
common length; and each of those is different per task, per checkpoint, and per
release of ``transformers``.

A ``transformers.pipeline`` already knows all of it — which preprocessors a task
loads, how to turn its inputs into model inputs, and how to collate them — so
this module leans on the pipeline rather than re-deriving any of it:

* **Loading**: ``pipeline(model=repo_id, ...)`` infers the preprocessors the task
  needs; the task's pipeline class says which those are through its ``_load_*``
  flags. The lazy meta build is the exception — ``pipeline()`` can't
  ``from_config`` a model, so the meta model is built here and handed to it.
* **Input**: each invoke goes through the task's own ``preprocess`` (with its own
  ``_sanitize_parameters`` splitting preprocess from forward kwargs), and the
  per-invoke encodings are padded together by `processing.collate`.
* **Padding**: which side to pad is the model's business, not the task's, so it
  follows `TransformersModel._is_causal` — decoders left-pad and get
  mask-derived ``position_ids``; encoders and encoder-decoders pad right.

Three ways in, and the difference matters:

* ``trace`` runs **one forward**. Its input is assembled here, so it accepts what
  the model accepts: text, token ids, a tensor, or an encoding.
* ``generate`` generates **through the model** and returns token ids. It takes the
  same inputs a forward does (assembled here) and generates with the checkpoint's
  own settings, not the ``task_specific_params`` a pipeline folds in.
* ``pipe`` runs **the whole pipeline**, which preprocesses and collates its own
  text — so it takes what that pipeline takes — and returns what the pipeline
  postprocesses to (decoded text, labels, ...).

Some inputs can't be padded into a batch at all — a raw feature tensor, or a
multimodal encoding — so a lone invoke carries them straight to the model, and
asking to batch several of them is refused rather than silently mangled.

A chunked task splits one input into several encodings, each forwarded on its
own: ``token-classification`` past the model's length limit, one entailment pair
per candidate label in ``zero-shot-classification``, a long recording's windows
in ``automatic-speech-recognition``. Those become rows of the trace's one
forward — which is what the pipeline does at a ``batch_size`` of its chunk count
— so a read inside the block sees one row per chunk, in the order the task
yields them. A chunked invoke is the whole batch: the row count is the task's to
decide, and the trace counts one row per invoke, so batching it against another
invoke is refused rather than served the wrong rows.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

import warnings

import torch

from .. import NNsightDeprecationWarning
from ..intervention.envoy import Envoy, traceable
from . import peft, processing
from .huggingface import HuggingFaceModel

if TYPE_CHECKING:
    from transformers import (
        BaseImageProcessor,
        FeatureExtractionMixin,
        Pipeline,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )


_PREPROCESSORS = ("tokenizer", "image_processor", "feature_extractor", "processor")

# Attribute -> persistent id: for a remote request these are referenced by id
# rather than serialized, and resolved to the actor's live object server-side (see
# _remoteable_persistent_objects / __getstate__). The pipeline is included so a
# deserialized model's `self.pipeline` (used by generate) resolves to the server's
# real pipeline instead of being dropped.
_PERSISTENT = {
    "tokenizer": "Tokenizer",
    "processor": "Processor",
    "image_processor": "ImageProcessor",
    "feature_extractor": "FeatureExtractor",
    "pipeline": "Pipeline",
}

# Kwargs the meta build forwards to AutoModel.from_config itself. Everything that
# belongs on the config reaches it through AutoConfig instead (see _load_meta);
# the meta build reconstructs structure only, so weight/placement kwargs
# (device_map, max_memory, ...) are dropped — they're meaningless on meta tensors,
# and from_config forwards unknown kwargs to the model __init__, which rejects them.
# trust_remote_code is the important one: it decides which class (and thus module
# tree) is built, so the client's meta model matches the server's real model.
_META_MODEL_KWARGS = ("trust_remote_code", "torch_dtype", "dtype", "attn_implementation")

# Architecture-class suffix -> pipeline task, for inferring a task from a pre-loaded
# module (the pipeline factory can only infer a task from a repo-id string).
_ARCH_TASK = {
    "ForCausalLM": "text-generation",
    "ForConditionalGeneration": "text-generation",
    "ForMaskedLM": "fill-mask",
    "ForSequenceClassification": "text-classification",
    "ForTokenClassification": "token-classification",
    "ForImageClassification": "image-classification",
    "ForImageTextToText": "image-text-to-text",
}


def _infer_task(module: torch.nn.Module) -> str:
    """Infer a pipeline task from a pre-loaded module.

    A generative model (``can_generate()`` — covers ``*ForCausalLM``,
    ``*LMHeadModel``, ...) is text-generation; otherwise match the architecture
    class-name suffix (``*ForMaskedLM`` -> fill-mask, ...).
    """
    if getattr(module, "can_generate", lambda: False)():
        return "text-generation"
    names = getattr(module.config, "architectures", None) or [type(module).__name__]
    for name in names:
        for suffix, task in _ARCH_TASK.items():
            if name.endswith(suffix):
                return task
    raise ValueError(
        f"Could not infer a pipeline task for a pre-loaded {type(module).__name__}; "
        "pass task=... explicitly (e.g. TransformersModel(model, task='text-generation'))."
    )


def _split_pipeline_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Split kwargs into (top-level pipeline args, model_kwargs) for ``pipeline()``.

    Names the pipeline factory declares stay top-level so a cross-cutting arg like
    ``trust_remote_code`` reaches the model *and* the config/tokenizer load; the
    rest are from_pretrained-only (e.g. ``max_memory``) and go through
    ``model_kwargs``. Passed top-level, an unrecognized name is stashed in the
    pipeline's ``_forward_params`` and forwarded to ``model.generate``, which
    rejects it ("model_kwargs not used by the model: ['max_memory']").
    """
    import inspect

    from transformers import pipeline

    factory_params = {
        name
        for name, parameter in inspect.signature(pipeline).parameters.items()
        if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
    }
    top_level = {k: v for k, v in kwargs.items() if k in factory_params}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in factory_params}
    return top_level, model_kwargs


class WrapperModule(torch.nn.Module):
    """Identity module: returns its input unchanged.

    Lets nnsight expose a value that isn't produced by a real submodule — the
    value is passed *through* this module so it is served at the module's
    ``.output``.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return args[0] if len(args) == 1 else args


class Generator(WrapperModule):
    """Passthrough for the generation output.

    Generation output is passed through this module so it is readable/editable at
    ``model.generator.output`` inside a trace. Its [`Streamer`][nnsight.modeling.transformers.Generator.Streamer] submodule
    receives tokens as they are decoded (HuggingFace's ``streamer`` protocol), so
    ``model.generator.streamer.output`` gives per-step token access.

    Reading the finished ids through ``model.generator.output`` is deprecated —
    ``generate`` returns them, so use ``tracer.result`` instead. The module stays
    for the per-step ``streamer`` access, which ``tracer.result`` has no equivalent
    for.
    """

    class Streamer(WrapperModule):
        """Receives generated tokens during decoding via ``put`` / ``end``."""

        def put(self, value: Any) -> Any:
            return self(value)

        def end(self) -> None:
            pass

    def __init__(self) -> None:
        super().__init__()
        self.streamer = Generator.Streamer()


_GENERATOR_OUTPUT_DEPRECATED = (
    "model.generator.output is deprecated; use tracer.result instead "
    "(model.generator.streamer.output still gives per-step tokens)."
)


class GeneratorEnvoy(Envoy):
    """The envoy for `Generator`, whose ``.output`` is deprecated.

    ``model.generator.output`` is the only served value in nnsight that is
    deprecated rather than removed, so the warning lives on the envoy of the one
    module that has it — the rest of the tree keeps the plain `Envoy`.
    """

    @property
    def output(self) -> Any:
        """Deprecated: the finished generated ids — read ``tracer.result``.

        A plain property wrapping `Envoy.output`, not an `eproperty` of its own:
        the warning has to reach the user *before* the read parks the worker, and
        an eproperty's preprocess runs only once the value has been served.
        """
        warnings.warn(
            _GENERATOR_OUTPUT_DEPRECATED, NNsightDeprecationWarning, stacklevel=2
        )
        return Envoy.output.__get__(self)

    @output.setter
    def output(self, value: Any) -> None:
        warnings.warn(
            _GENERATOR_OUTPUT_DEPRECATED, NNsightDeprecationWarning, stacklevel=2
        )
        Envoy.output.__set__(self, value)


class TransformersModel(HuggingFaceModel):
    """A model backed by a ``transformers.pipeline``, for any of its tasks.

    See the module docstring for what the pipeline is leaned on for. ``task`` picks
    the pipeline (inferred from the checkpoint when unset). There are three ways to
    run it: `trace` runs one forward, `generate` generates through the
    model and returns token ids, and [`pipe`][nnsight.modeling.transformers.TransformersModel.pipe] runs the whole pipeline and
    returns what it postprocesses to (decoded text, labels, ...).

    The pipeline and its preprocessors are exposed as attributes, so the
    tokenizer that will actually be used is ``model.tokenizer``. Which of them a
    task loads varies — a text task has a ``tokenizer`` and no
    ``image_processor``, a multimodal one has a ``processor`` — so any of them
    may be ``None``. Passing one in adopts it instead of loading it.

    Attributes:
        pipeline: The task's pipeline. Owns the model and its preprocessors.
        tokenizer: The tokenizer, for a task that has one.
        processor: The processor, for a multimodal task.
        image_processor: The image processor, for a vision task.
        feature_extractor: The feature extractor, for an audio task.
        generator: The module generated ids are passed through. Reading them at
            ``model.generator.output`` is deprecated (use ``tracer.result``); it
            remains for per-step access at ``.streamer.output``.
    """

    pipeline: Optional["Pipeline"]
    tokenizer: Optional["PreTrainedTokenizerBase"]
    processor: Optional["ProcessorMixin"]
    image_processor: Optional["BaseImageProcessor"]
    feature_extractor: Optional["FeatureExtractionMixin"]

    def __init__(
        self,
        repo_id: Any,
        *args: Any,
        task: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        image_processor: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        peft: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.pipeline = None
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = image_processor
        self.feature_extractor = feature_extractor
        # HuggingFace repo id of a PEFT adapter to apply on top of the base
        # model. Applied at load time (below); server-side it can be swapped
        # per request via _remoteable_set_env without redeploying the base.
        self.peft = peft

        # A sharded module called ad hoc — a logit lens — deals in one rank's
        # slices either side, where the caller is holding whole tensors.
        # `TPEnvoy` corrects that, but it is keyed by module *type* (every Linear
        # and Embedding in the tree) because the style that decides the
        # correction is stamped on the instance at load rather than carried by a
        # class. So it goes on only when this construction is actually going to
        # shard something; an ordinary model keeps the plain `Envoy` it always
        # had. A caller passing `envoys` of their own replaces it wholesale.
        from .tp.envoys import tp_envoys, wants_tensor_parallel

        if wants_tensor_parallel(repo_id, kwargs):
            kwargs.setdefault("envoys", tp_envoys())

        super().__init__(repo_id, *args, **kwargs)

        # A standalone module (not part of the HF model) that generation output is
        # passed through, so per-step tokens reach `model.generator.streamer.output`
        # (reading the finished ids at `model.generator.output` is deprecated in
        # favor of `tracer.result`). Added to `_children` so it shows in the tree;
        # `_update` (dispatch) and `_remoteable_set_env` (PEFT rebind) both preserve
        # standalone children like this one.
        self.generator = GeneratorEnvoy(
            Generator(),
            path=f"{self.path}.generator",
            interleaver=self.interleaver,
            parent=self,
        )
        self._children.append(self.generator)

    # -- loading -------------------------------------------------------------

    def _preprocessor_sources(self) -> dict:
        # Feed pipeline a source for each preprocessor: the provided object, or
        # the repo id for it to load (needed for the meta model, which has no
        # path to infer from). Only the preprocessors the task's pipeline actually
        # loads are sourced — passing a stray tokenizer source to a processor-based
        # (multimodal) pipeline, for instance, makes it reject the string.
        needed = self._loaded_preprocessors()
        sources = {
            attr: getattr(self, attr) or self.repo_id
            for attr in _PREPROCESSORS
            if getattr(self, attr) is not None or attr in needed
        }
        # The audio pipelines look for a CTC decoder when the feature extractor
        # comes in as a repo id, under a model name they derive from the model
        # argument -- which is None when the model is pre-built (the meta path),
        # so they fetch `huggingface.co/None/...` and die. Hand them the object
        # instead; a feature extractor is a JSON config, no weights, so building
        # it costs nothing on the path that exists to avoid loading weights.
        if isinstance(sources.get("feature_extractor"), str):
            from transformers import AutoFeatureExtractor

            sources["feature_extractor"] = AutoFeatureExtractor.from_pretrained(
                sources["feature_extractor"], revision=self.revision
            )
        return sources

    def _loaded_preprocessors(self) -> set:
        # Which of tokenizer/image_processor/feature_extractor/processor the task's
        # pipeline class loads, read from its _load_* class flags (e.g. text-
        # generation loads a tokenizer; image-text-to-text loads a processor).
        from transformers.pipelines import check_task

        flags = {
            "tokenizer": "_load_tokenizer",
            "image_processor": "_load_image_processor",
            "feature_extractor": "_load_feature_extractor",
            "processor": "_load_processor",
        }
        try:
            _, targeted, _ = check_task(self.task)
            impl = targeted["impl"]
        except Exception:  # noqa: BLE001 - unknown/unset task: source them all
            return set(_PREPROCESSORS)
        return {attr for attr, flag in flags.items() if getattr(impl, flag, False)}

    def _sync(self) -> None:
        # Adopt the pipeline's task and preprocessors. Slots the task didn't
        # load come back as the raw repo-id string, so null those.
        self.task = self.pipeline.task
        for attr in _PREPROCESSORS:
            value = getattr(self.pipeline, attr, None)
            setattr(self, attr, None if isinstance(value, str) else value)
        self._configure_tokenizer()

    def _configure_tokenizer(self) -> None:
        # Pad with EOS when there's no pad token. Left-pad only for causal decoders,
        # so a batched trace/generation aligns the last real token at the right edge
        # (``output[:, -1]`` is every row's real last token); encoder tasks keep
        # their default (right) padding. An encoder-decoder is set to the right
        # explicitly: the text-generation pipeline it loads under left-pads every
        # model it wraps, which shifts the absolute positions of BART's encoder.
        if self.tokenizer is None:
            return
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self._is_causal():
            self.tokenizer.padding_side = "left"
        elif self._is_encoder_decoder():
            self.tokenizer.padding_side = "right"

    def _is_causal(self) -> bool:
        # A decoder-only generative model (GPT-2, Llava, ...) — as opposed to an
        # encoder (BERT) or encoder-decoder (T5). Decides left-padding and the
        # left-pad position_ids correction.
        model = getattr(self.pipeline, "model", None)
        return (
            model is not None
            and model.can_generate()
            and not self._is_encoder_decoder()
        )

    def _is_encoder_decoder(self) -> bool:
        model = getattr(self.pipeline, "model", None)
        return bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))

    def _load_meta(self, repo_id: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
        from transformers import AutoConfig, pipeline
        from transformers.pipelines import check_task, get_task

        from .quantization import resolve_load_kwargs

        # A quantization name in `dtype` becomes the compute dtype here and no
        # quantizer config: there are no weights on meta to quantize. Done before
        # the filter below, not after, so an explicit compute dtype (which is not
        # an architecture kwarg and would be dropped) still reaches the build.
        kwargs = resolve_load_kwargs(kwargs, quantize=False)

        # Only the kwargs in _META_MODEL_KWARGS reach from_config;
        # AutoConfig.from_pretrained tolerates extras but from_config does not,
        # and placement kwargs don't apply to meta tensors.
        #
        # `dtype="auto"` is dropped: it means "read the dtype off the checkpoint
        # weights", which only from_pretrained can do — there are none on meta.
        # from_config resolves a string dtype with `getattr(torch, dtype)`, so
        # leaving it in raises AttributeError. Dropping it here also keeps it off
        # AutoConfig, which would otherwise store the literal "auto" as
        # `config.dtype` and hand from_config the same string by default. What
        # remains is the checkpoint's own declared dtype — which is what "auto"
        # resolves to first anyway.
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if not (k in ("dtype", "torch_dtype") and v == "auto")
        }
        arch = {k: v for k, v in kwargs.items() if k in _META_MODEL_KWARGS}

        # pipeline can't from_config, so resolve the task's model classes and
        # build the meta model ourselves, then wrap it in a meta pipeline.
        self.task = self.task or get_task(repo_id)
        _, targeted, _ = check_task(self.task)

        # The config takes every kwarg, and keeps the ones it owns, the same split
        # from_pretrained makes (`return_unused_kwargs=True`): the implementation
        # selectors (`attn_implementation`, `experts_implementation`, and whatever
        # transformers adds next), `dtype`, and overrides of config attributes. So
        # the meta model is built from the same config the real load will use.
        # The rest (placement, model __init__ kwargs) comes back unused and is
        # dropped. `quantization_config` is held back: from_pretrained takes it
        # before the config does and merges it with a pre-quantized checkpoint's
        # own, so letting the config claim it would overwrite that on meta.
        config, _ = AutoConfig.from_pretrained(
            repo_id,
            revision=self.revision,
            return_unused_kwargs=True,
            **{k: v for k, v in kwargs.items() if k != "quantization_config"},
        )

        error = None
        for auto in targeted["pt"]:
            try:
                model = auto.from_config(config, **arch)
                break
            except Exception as exception:  # noqa: BLE001 - try next candidate
                error = exception
        else:
            raise error

        if self.peft is not None:
            # Read only the adapter's config (adapter_config.json) and graft the
            # adapter modules onto the meta model, so the meta architecture — and
            # thus the module paths a remote request references — matches the
            # adapted model the server runs. No adapter weights are loaded here.
            peft.attach(model, self.peft, weights=False)

        # The model is pre-built, so the meta pipeline only loads preprocessors;
        # pass the pipeline-recognized arch kwargs (e.g. trust_remote_code) so a
        # custom tokenizer loads correctly.
        top_level, _ = _split_pipeline_kwargs(arch)
        self.pipeline = pipeline(
            self.task,
            model=model,
            device="meta",
            **self._preprocessor_sources(),
            **top_level,
        )
        self._sync()
        return model

    def _load(self, repo_id: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
        from transformers import pipeline

        from .quantization import resolve_load_kwargs

        # Before the split, and before the pipeline fetches anything. This path
        # does not reach the base's `_load`, so the check has to be repeated
        # here -- the tensor-parallel server loads through *this* class.
        self._refuse_impossible_tp(repo_id, kwargs)

        # Also before the split: `dtype` is a pipeline-factory argument, so a
        # quantization name left in it would be handed to `pipeline()` rather
        # than to the quantizer. `quantization_config` is not a factory argument
        # and lands in `model_kwargs`, which is where from_pretrained wants it.
        kwargs = resolve_load_kwargs(kwargs)

        top_level, model_kwargs = _split_pipeline_kwargs(kwargs)
        # The pipeline loads the model and infers every preprocessor; only
        # forward the ones the user explicitly supplied.
        provided = {
            attr: getattr(self, attr)
            for attr in _PREPROCESSORS
            if getattr(self, attr) is not None
        }
        self.pipeline = pipeline(
            self.task,
            model=repo_id,
            revision=self.revision,
            **provided,
            **top_level,
            model_kwargs=model_kwargs,
        )
        return self._finalize_pipeline()

    def _wrap(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        from transformers import pipeline

        top_level, _ = _split_pipeline_kwargs(kwargs)
        # The pipeline factory can't infer the task or the preprocessors from a
        # module instance, so infer the task and source the preprocessors from
        # what was passed in or the model's name_or_path (captured as
        # self.repo_id). The class-name guess can differ from the Hub's
        # pipeline_tag — which is what a repo-id construction infers into the
        # remote model key — but a pre-loaded module is a *local* model: its
        # weights are already in hand, so its key never has to match a
        # deployment. Local inference keeps this path offline.
        if self.task is None:
            self.task = _infer_task(module)
        self.pipeline = pipeline(
            self.task, model=module, **self._preprocessor_sources(), **top_level
        )
        return self._finalize_pipeline()

    def _finalize_pipeline(self) -> torch.nn.Module:
        if self.peft is not None:
            # The pipeline loaded the base weights; attach the adapter with its
            # real weights so the dispatched model runs with the adapter applied.
            peft.attach(self.pipeline.model, self.peft)
        self._sync()
        return self.pipeline.model

    # -- running -------------------------------------------------------------

    def trace(self, *inputs: Any, fn: Any = None, **kwargs: Any):
        if fn is None:
            fn = self._call
        return super().trace(*inputs, fn=fn, **kwargs)

    def scan(self, *inputs: Any, fn: Any = None, **kwargs: Any):
        # Same forward as trace (so a string prompt is tokenized by _call),
        # but under fake tensors — see Meta.scan.
        if fn is None:
            fn = self._call
        return super().scan(*inputs, fn=fn, **kwargs)

    @traceable
    def generate(self, *inputs: Any, **kwargs: Any) -> Any:
        """Generate through the model, returning the generated token ids.

        ``with model.generate(...):`` traces the generation, so the block's
        interventions run against every forward the decode loop makes — use
        ``tracer.iter`` to target a particular step. Calling it directly just
        generates. The output is the whole prompt plus completion as token ids.

        Generating goes through the model, not the task's pipeline (see [`pipe`][nnsight.modeling.transformers.TransformersModel.pipe]
        for that): the model takes the same inputs a forward does — text, token ids,
        a tensor, or an encoding — and generates the way calling it would, with the
        checkpoint's own settings rather than the ``task_specific_params`` a pipeline
        would fold in. Read the ids off ``tracer.result``; they also pass through
        [`generator`][nnsight.modeling.transformers.TransformersModel.generator], whose ``model.generator.streamer.output`` gives per-step
        access (reading the finished ids at ``model.generator.output`` is deprecated
        in favor of ``tracer.result``).

        Examples:
            >>> model = TransformersModel("openai-community/gpt2", dispatch=True)
            >>> with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
            ...     ids = tracer.result.save()
            >>> print(model.tokenizer.batch_decode(ids))

        Args:
            *inputs: What to generate from — the same forms `trace` takes.
            **kwargs: Passed to the model's ``generate``, e.g. ``max_new_tokens``.
                ``streamer`` defaults to this model's (except under beam search,
                which transformers refuses a streamer for); pass it to override.

        Returns:
            The generated token ids, as a ``[batch, seq]`` tensor.
        """
        # transformers refuses any streamer under beam search, and this one is
        # nnsight's, not the caller's -- so only inject it when the run is
        # single-beam. The beams can come from a passed generation_config or from
        # the checkpoint's own, where unset reads as None rather than 1.
        config = kwargs.get("generation_config") or getattr(
            self.pipeline.model, "generation_config", None
        )
        num_beams = kwargs.get("num_beams") or getattr(config, "num_beams", None) or 1
        if num_beams == 1:
            kwargs.setdefault("streamer", self.generator.streamer._module)
        output = self.pipeline.model.generate(*inputs, **kwargs)
        # Pass the output through the generator module so a worker parked on
        # `model.generator.output` receives it (and can edit it). hook=True fires the
        # module's hooks even mid-interleave so that `.output` is observable.
        return self.generator(output, hook=True)

    @traceable
    def pipe(self, *inputs: Any, **kwargs: Any) -> Any:
        """Run the task's pipeline end to end, returning what it postprocesses to.

        Where `generate` goes through the model and returns token ids, this
        runs the whole pipeline — decoded-text records for text-generation, labels
        for a classifier, and so on — the pipeline tokenizing and collating its own
        input. Traced like the others: the block sees every forward the pipeline
        makes.

        Examples:
            >>> model = TransformersModel("openai-community/gpt2", dispatch=True)
            >>> with model.pipe("The Eiffel Tower is in", max_new_tokens=3) as tracer:
            ...     out = tracer.result.save()
            >>> print(out[0]["generated_text"])

        Args:
            *inputs: Inputs for the task's pipeline — text, chat messages, images.
            **kwargs: Passed to the pipeline, e.g. ``max_new_tokens``.

        Returns:
            The task pipeline's postprocessed output.
        """
        # Dispatch is handled by interleave when tracing.
        return self.pipeline(*inputs, **kwargs)

    # -- remote --------------------------------------------------------------

    def _remoteable_model_key(self) -> str:
        # The task is part of the model's remote identity: two tasks over one
        # checkpoint can load different architecture classes (ForCausalLM vs
        # ForSequenceClassification), so they are different deployments, and the
        # server must rebuild the pipeline the client traced rather than
        # re-infer one from the Hub. Always the resolved task, never null — an
        # unset task is inferred by the meta build before any key is minted.
        # The alias table, not check_task's full normalization: an alias names
        # the identical pipeline ("sentiment-analysis" is "text-classification"),
        # so two spellings must not become two deployments — but check_task
        # would also collapse "translation_en_to_fr" to bare "translation",
        # which names a *different* pipeline configuration.
        from transformers.pipelines import TASK_ALIASES

        data = json.loads(super()._remoteable_model_key())
        data["task"] = TASK_ALIASES.get(self.task, self.task)
        return json.dumps(data)

    def _remoteable_persistent_objects(self) -> dict:
        objects = super()._remoteable_persistent_objects()
        for attr, pid in _PERSISTENT.items():
            value = getattr(self, attr)
            if value is not None:
                objects[pid] = value
        return objects

    def _remoteable_get_env(self) -> dict:
        """The per-request environment this model wants applied server-side.

        Returned client-side and carried with a remote request; the server
        applies it via `_remoteable_set_env` before running. Only the PEFT
        adapter is transported — the base model is identified by the model key.
        """
        return {} if self.peft is None else {"peft": self.peft}

    def _remoteable_set_env(self, env: Optional[dict]) -> None:
        """Apply a per-request environment on the server side.

        Only the PEFT adapter travels — the base model is identified by the
        model key. See `peft.swap` for the transitions.
        """
        peft.swap(self, env.get("peft") if env else None)

    def load_adapter(self, peft_id: Optional[str]) -> None:
        """Attach a PEFT adapter — the post-hoc form of the ``peft=`` kwarg.

        Call again with a different id to swap adapters, or with ``None`` to
        remove the current one (see `peft.swap` for the transitions).

        Works before dispatch: only the adapter's config is grafted onto the
        meta module, so the tree gains the adapter's modules — and remote
        module paths match — without loading any weights (safetensors cannot
        load onto the meta device). The adapter's real weights arrive with the
        base's at dispatch. That deferral means a wrong adapter id fails here
        (the config is read now), but an adapter that doesn't match the base
        is refused only at dispatch, where the weights meet.

        This shadows ``PreTrainedModel.load_adapter``, which the envoy would
        otherwise reach by fallthrough — that one fails on a meta model and
        changes the module structure without the envoy tree noticing. An
        adapter attached here also travels with remote requests
        (`_remoteable_get_env`).

        Args:
            peft_id: The adapter's repo id or local path, or ``None`` to
                remove the current adapter.
        """
        if self.dispatched:
            peft.swap(self, peft_id)
            return
        if peft_id == self.peft:
            return
        from .mixins.meta import MetaDevice

        self.peft = peft_id
        # Rebuild the meta skeleton exactly as construction would have built it
        # with this adapter: `_load_meta` reads `self.peft` and grafts the
        # adapter's architecture from its config alone.
        with MetaDevice():
            module = self._load_meta(*self.args, **self.kwargs)
        self._update(module)
        if self.pipeline is not None:
            self.pipeline.model = module

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        # Reference the pipeline and preprocessors by persistent id instead of
        # serializing them; the server resolves each to its live object. The
        # pipeline stays in state (tagged) so generate's `self.pipeline` resolves.
        for attr, pid in _PERSISTENT.items():
            value = getattr(self, attr)
            if value is not None:
                value._persistent_id = pid
        return state

    # -- forward -------------------------------------------------------------

    def _call(self, *inputs: Any, **kwargs: Any) -> Any:
        preprocessor = (
            self.processor
            or self.tokenizer
            or self.image_processor
            or self.feature_extractor
        )
        if preprocessor is not None and inputs and isinstance(inputs[0], (str, list)):
            # BatchEncoding/BatchFeature: .to(device) moves tensors, ** unpacks.
            prepared = preprocessor(*inputs, return_tensors="pt").to(
                next(self._module.parameters()).device
            )
            return self._module(**prepared, **kwargs)
        return self._module(*inputs, **kwargs)

    # -- batching ------------------------------------------------------------
    #
    # The two hooks a trace batches through. What they do with an input — the
    # processor, tokenization, padding, positions — is in `processing`.

    def _batch_size(self, *inputs: Any, **kwargs: Any) -> int:
        """Number of batch rows an invoke's input contributes (see `processing.batch_size`)."""
        return processing.batch_size(inputs, kwargs)

    def _batch(self, invokes: list, fn: Any) -> tuple:
        """Combine invokes into one input for ``fn``.

        ``pipe`` runs the whole pipeline, so text prompts are handed to it as a list
        with ``batch_size`` (it preprocesses and collates them itself). ``generate``
        and ``trace`` run the model, so their input is assembled into model inputs
        (`processing.batch_forward`).
        """
        if getattr(fn, "__name__", None) == "pipe":
            return processing.batch_pipe(invokes)
        return processing.batch_forward(self, invokes)
