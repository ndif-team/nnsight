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
  per-invoke encodings are padded together by the pipeline's ``pad_collate_fn``.
* **Padding**: which side to pad is the model's business, not the task's, so it
  follows `TransformersModel._is_causal` — decoders left-pad and get
  mask-derived ``position_ids``; encoders keep right padding.

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

from typing import TYPE_CHECKING, Any, Optional

import sys
import warnings

import torch
from torch._guards import detect_fake_mode

from .. import NNsightDeprecationWarning
from ..intervention.envoy import Envoy, traceable
from .huggingface import HuggingFaceModel

if TYPE_CHECKING:
    from transformers import (
        BaseImageProcessor,
        FeatureExtractionMixin,
        Pipeline,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )


def _import_peft():
    """Import ``peft`` on demand (it's an optional dependency).

    Only needed when a ``peft=<repo_id>`` adapter is requested, so importing it
    lazily keeps nnsight usable without peft installed.
    """
    try:
        import peft
    except ImportError as error:
        raise ImportError(
            "Using `peft=<repo_id>` requires the optional `peft` package, which "
            "is not installed. Install it with `pip install peft`."
        ) from error
    return peft

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

# Architecture-shaping kwargs the meta build forwards to AutoModel.from_config.
# The meta build reconstructs structure only, so weight/placement kwargs
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
    "ForQuestionAnswering": "question-answering",
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


def _refuse_noop_peft(peft_id: str, caught: list) -> None:
    """Turn peft's "missing adapter keys" warning into an error.

    peft places adapter weights by **name** and drops the ones it cannot match. As
    ``lora_B`` initialises to zeros, an adapter whose weights did not land is
    exactly the identity -- the model behaves like the base checkpoint, so a
    base-vs-adapter comparison silently becomes base-vs-base with every number in
    it plausible. peft warns about this precisely because ``from_pretrained``
    cannot return its load result (see `PeftModel.from_pretrained`), but a warning
    is easy to miss in a long load, and by the time it matters the run is finished.

    Only the mismatch warns: an adapter whose keys all place -- including a freshly
    initialised one whose ``lora_B`` is legitimately still zero -- produces none.
    """
    missing = [
        warning
        for warning in caught
        if "missing adapter keys" in str(warning.message).lower()
    ]
    if not missing:
        return

    raise ValueError(
        f"The PEFT adapter {peft_id!r} did not attach: peft could not match its "
        "weights to this model's modules by name, dropped them, and left the "
        "adapter at its zero initialisation -- so it is a no-op and the model "
        "would behave exactly like the base checkpoint. The usual cause is a "
        "`task=` that builds a different architecture than the adapter was trained "
        "against: e.g. task='text-generation' where the adapter targets a "
        "multimodal config, which needs task='image-text-to-text'. peft reported: "
        f"{str(missing[0].message)[:400]}"
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
            Generator(), path=f"{self.path}.generator", interleaver=self.interleaver
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
        return {
            attr: getattr(self, attr) or self.repo_id
            for attr in _PREPROCESSORS
            if getattr(self, attr) is not None or attr in needed
        }

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
        # their default (right) padding.
        if self.tokenizer is None:
            return
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self._is_causal():
            self.tokenizer.padding_side = "left"

    def _is_causal(self) -> bool:
        # A decoder-only generative model (GPT-2, Llava, ...) — as opposed to an
        # encoder (BERT) or encoder-decoder (T5). Decides left-padding and the
        # left-pad position_ids correction.
        model = getattr(self.pipeline, "model", None)
        config = getattr(model, "config", None)
        return (
            model is not None
            and model.can_generate()
            and not getattr(config, "is_encoder_decoder", False)
        )

    def _load_meta(self, repo_id: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
        from transformers import AutoConfig, pipeline
        from transformers.pipelines import check_task, get_task

        from .quantization import resolve_load_kwargs

        # A quantization name in `dtype` becomes the compute dtype here and no
        # quantizer config: there are no weights on meta to quantize. Done before
        # the filter below, not after, so an explicit compute dtype (which is not
        # an architecture kwarg and would be dropped) still reaches the build.
        kwargs = resolve_load_kwargs(kwargs, quantize=False)

        # Only architecture-shaping kwargs reach the meta build (see
        # _META_MODEL_KWARGS); AutoConfig.from_pretrained tolerates extras but
        # from_config does not, and placement kwargs don't apply to meta tensors.
        #
        # `dtype="auto"` is dropped: it means "read the dtype off the checkpoint
        # weights", which only from_pretrained can do — there are none on meta.
        # from_config resolves a string dtype with `getattr(torch, dtype)`, so
        # leaving it in raises AttributeError. Dropping it here also keeps it off
        # AutoConfig, which would otherwise store the literal "auto" as
        # `config.dtype` and hand from_config the same string by default. What
        # remains is the checkpoint's own declared dtype — which is what "auto"
        # resolves to first anyway.
        arch = {
            k: v
            for k, v in kwargs.items()
            if k in _META_MODEL_KWARGS
            and not (k in ("dtype", "torch_dtype") and v == "auto")
        }

        # pipeline can't from_config, so resolve the task's model classes and
        # build the meta model ourselves, then wrap it in a meta pipeline.
        self.task = self.task or get_task(repo_id)
        _, targeted, _ = check_task(self.task)
        config = AutoConfig.from_pretrained(repo_id, revision=self.revision, **arch)

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
            peft = _import_peft()
            model = peft.get_peft_model(model, peft.PeftConfig.from_pretrained(self.peft))

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
        # self.repo_id).
        if self.task is None:
            self.task = _infer_task(module)
        self.pipeline = pipeline(
            self.task, model=module, **self._preprocessor_sources(), **top_level
        )
        return self._finalize_pipeline()

    def _finalize_pipeline(self) -> torch.nn.Module:
        if self.peft is not None:
            # The pipeline loaded the base weights; wrap them with the adapter's
            # real weights so the dispatched model runs with the adapter applied.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.pipeline.model = _import_peft().PeftModel.from_pretrained(
                    self.pipeline.model, self.peft
                )
            for warning in caught:
                warnings.warn(warning.message, warning.category)
            _refuse_noop_peft(self.peft, caught)
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
                ``streamer`` defaults to this model's; pass it to override.

        Returns:
            The generated token ids, as a ``[batch, seq]`` tensor.
        """
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

        Swaps the PEFT adapter to match ``env["peft"]``, rewrapping the loaded
        module only when the requested adapter differs from the current one so a
        repeat request pays nothing:

            current  requested  action
            -------  ---------  ------
            None     None       no-op
            None     X          load X
            X        X          no-op
            X        Y          unload X, load Y
            X        None       unload X
        """
        requested = env.get("peft") if env else None
        if requested == self.peft:
            return

        # Rebuild the Envoy tree around the new module in place: a wrap/unwrap
        # changes the module structure (adapter modules appear or disappear), so
        # re-init rather than _update, reusing this envoy's interleaver and rename
        # spec. Drop the previous tree's child-envoy attributes first — the new
        # structure has different top-level children, and __init__ resets
        # _children without clearing the stale attributes those children left.
        def rebind(module: torch.nn.Module) -> None:
            # Standalone children (whose module isn't part of the HF tree, e.g. the
            # generator) survive the swap: Envoy.__init__ builds _children only from
            # `module.named_children()`, so carry them across the re-init by name.
            submodules = set(self._module.modules())
            standalone = {
                name: value
                for name, value in self.__dict__.items()
                if isinstance(value, Envoy)
                and value is not self
                and value._module not in submodules
            }
            for name, value in list(self.__dict__.items()):
                if isinstance(value, Envoy) and value is not self:
                    del self.__dict__[name]
            Envoy.__init__(
                self, module, path=self.path, interleaver=self.interleaver, rename=self._rename
            )
            for name, child in standalone.items():
                self.__dict__[name] = child
                self._children.append(child)
            if self.pipeline is not None:
                self.pipeline.model = module

        if self.peft is not None:
            rebind(self._module.unload())
            # The module is the base checkpoint from here; keep `self.peft` honest
            # so a refused load below leaves this envoy self-consistent.
            self.peft = None

        if requested:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                adapted = _import_peft().PeftModel.from_pretrained(self._module, requested)
            for warning in caught:
                warnings.warn(warning.message, warning.category)
            # Check before rebinding, so a no-op adapter never becomes this envoy's
            # module. A swap is where this matters most: sweeping several adapters
            # over one loaded base is exactly the workload where every organism
            # silently collapsing to the base checkpoint looks like a real result.
            _refuse_noop_peft(requested, caught)
            rebind(adapted)

        self.peft = requested

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

    # Encoding keys that are purely text/token; an encoding carrying anything else
    # (pixel_values, input_features, ...) is multimodal and the model derives its
    # own positions, so the left-pad position_ids correction is skipped for it.
    _TEXT_KEYS = frozenset(
        {"input_ids", "attention_mask", "token_type_ids", "position_ids", "labels"}
    )

    # Non-text arguments a task's *processor* takes. An invoke naming any of these is
    # written in processor terms (``trace(prompt, images=[img])``) rather than model
    # terms (``trace(input_ids=...)``), so the processor has to run over it first.
    _PROCESSOR_MEDIA_KEYS = frozenset(
        {"images", "image", "audio", "audios", "videos", "video"}
    )
    # ``text`` rides along with the media keys but never triggers the processor path on
    # its own — text-only input is handled by the ordinary tokenization route.
    _PROCESSOR_KEYS = _PROCESSOR_MEDIA_KEYS | {"text"}

    def _batch_size(self, *inputs: Any, **kwargs: Any) -> int:
        """Number of batch rows an invoke's input contributes.

        Accepts every input format a forward takes: a string is one row, a list of
        strings one per prompt, a single token-id list one row, and a batch of
        token-id lists / a 2-D tensor / a pre-tokenized encoding one per leading
        entry. Multimodal data passed by keyword (``text=``/``images=`` to a VLM's
        generate) counts as one row. Zero rows means params only (e.g.
        ``max_new_tokens=``), so the trace expects ``invoke()`` blocks for the data.
        """
        value = inputs[0] if inputs else kwargs.get("input_ids")
        if value is not None:
            return self._num_rows(value)
        # A VLM generate passes its data by keyword; treat its presence as one row.
        if kwargs.get("text") is not None or any(
            kwargs.get(key) is not None
            for key in ("images", "pixel_values", "input_features")
        ):
            return 1
        return 0

    @staticmethod
    def _num_rows(value: Any) -> int:
        """The leading (row) dimension of one input value."""
        if isinstance(value, str):
            return 1
        if isinstance(value, torch.Tensor):
            return 1 if value.ndim <= 1 else value.shape[0]
        chats = TransformersModel._as_chats(value)
        if chats is not None:
            return len(chats)  # a chat conversation is one row, not one per message
        if isinstance(value, (list, tuple)):
            if not value:
                return 0
            # A flat list of token ids is a single sequence; a list of strings /
            # sub-sequences / tensors is one row per element.
            return 1 if isinstance(value[0], int) else len(value)
        if hasattr(value, "get") and value.get("input_ids") is not None:
            return TransformersModel._num_rows(value["input_ids"])
        # A lone non-text object (e.g. a PIL image) is a single row.
        return 1

    @staticmethod
    def _as_chats(data: Any, chat_cls: Optional[type] = None) -> Optional[list]:
        """Chat message(s) -> a list of ``Chat`` inputs (one per conversation).

        Mirrors the chat detection `Pipeline.__call__` does before preprocess,
        which calling `Pipeline.preprocess` directly would otherwise skip.
        Returns ``None`` when ``data`` isn't chat messages. ``chat_cls`` is the
        wrapper class to use — the pipeline's own when it defines one (see
        `_chat_cls`); the base ``Chat`` otherwise.
        """
        try:
            from transformers.pipelines.base import Chat, is_valid_message
        except ImportError:
            # transformers before the chat-pipeline refactor has neither
            # helper: treat everything as not-chat and let the pipeline's own
            # input handling take it from here.
            return None

        chat_cls = chat_cls or Chat
        if not isinstance(data, (list, tuple)) or not data:
            return None
        if is_valid_message(data[0]):
            return [chat_cls(list(data))]
        if all(
            isinstance(chat, (list, tuple)) and chat and is_valid_message(chat[0])
            for chat in data
        ):
            return [chat_cls(list(chat)) for chat in data]
        return None

    def _chat_cls(self) -> Optional[type]:
        """The ``Chat`` wrapper class this model's pipeline expects.

        Most chat pipelines isinstance-check ``transformers.pipelines.base.Chat``
        (or import it as a module attribute, which resolves to the same object),
        but ``any-to-any`` defines its *own* ``Chat`` and checks against that —
        a base-``Chat`` instance falls through to its raw-dict branch and fails
        on ``Chat.copy``. So a pipeline module that carries a ``Chat`` gets its
        own class.
        """
        module = sys.modules.get(type(self.pipeline).__module__)
        return getattr(module, "Chat", None)

    def _batch(self, invokes: list, fn: Any) -> tuple:
        """Combine invokes into one input for ``fn``.

        ``pipe`` runs the whole pipeline, so text prompts are handed to it as a list
        with ``batch_size`` (it preprocesses and collates them itself). ``generate``
        and ``trace`` run the model, so their input is assembled into model inputs
        here (see `_batch_forward`): each invoke's text/image is turned into
        model inputs by `Pipeline.preprocess` and the per-invoke encodings are
        padded together by the pipeline's own ``pad_collate_fn``; pre-tokenized ids
        and raw feature tensors bypass preprocessing.
        """
        name = getattr(fn, "__name__", None)
        if name == "pipe":
            return self._batch_pipe(invokes)
        if name == "generate":
            return self._batch_generate(invokes)
        return self._batch_forward(invokes)

    def _batch_generate(self, invokes: list) -> tuple:
        """Assemble invokes into model inputs for ``generate``.

        Generating through the model takes the same model inputs a forward does, so
        this is `_batch_forward` — overridden by models whose generate input
        needs different handling (a VLM runs its processor first).
        """
        return self._batch_forward(invokes)

    def _batch_pipe(self, invokes: list) -> tuple:
        """Hand text prompts to the pipeline, batched with ``batch_size``.

        A single non-text payload (a VLM's ``text=``/``images=`` keywords) is handed
        to the pipeline as-is; only string prompts are combined into a batch.
        """
        prompts: list = []
        forward: dict = {}
        passthrough = None
        for inputs, kwargs in invokes:
            data = inputs[0] if inputs else None
            if isinstance(data, str):
                prompts.append(data)
            elif isinstance(data, (list, tuple)) and data and isinstance(data[0], str):
                prompts.extend(data)
            else:
                passthrough = (inputs, kwargs)
                continue
            forward.update(kwargs)

        if passthrough is not None:
            if len(invokes) > 1:
                raise NotImplementedError(
                    "Batching multimodal generate inputs isn't supported; pass a "
                    "single text/images payload."
                )
            return passthrough

        # A single prompt keeps the pipeline's scalar-input output shape; a real
        # batch goes as a list with batch_size so it runs as one batch.
        if len(prompts) == 1:
            return (prompts[0],), forward
        return (prompts,), {**forward, "batch_size": len(prompts)}

    def _batch_forward(self, invokes: list) -> tuple:
        """Preprocess each invoke and pad the results into one forward input.

        Runs on CPU; interleave() moves inputs to the model's device after the
        (possibly lazy) dispatch, so don't touch device here.
        """
        items: list = []
        forward: dict = {}
        for inputs, kwargs in invokes:
            data = inputs[0] if inputs else None
            rows, forward_kwargs = self._preprocess_invoke(data, kwargs)
            if rows is None:
                # A raw feature tensor / multimodal encoding can't be padded into an
                # input_ids batch — pass a lone invoke straight to the model. An
                # encoding (positional or via kwargs) is unpacked as keyword inputs.
                if len(invokes) > 1:
                    raise NotImplementedError(
                        "Can't batch these inputs; pass text or token ids."
                    )
                if data is not None and hasattr(data, "keys"):
                    return tuple(), {**dict(data), **kwargs}
                # `forward_kwargs` is `kwargs` for a plain opaque input, and the
                # processor's encoding when the invoke was written in processor terms.
                if forward_kwargs is not kwargs:
                    return tuple(), forward_kwargs
                return inputs, kwargs
            # A chunked task decides its own row count, and the batcher counted
            # this invoke's input as its own rows before preprocessing — so with
            # another invoke in the batch every group after this one names rows
            # that belong to someone else, and each invoke's reads and edits land
            # on the wrong ones. Silent, so refuse it.
            if len(invokes) > 1 and len(rows) != self._batch_size(*inputs, **kwargs):
                raise NotImplementedError(
                    f"task={self.task!r} splits this invoke into {len(rows)} forward "
                    "rows, and a batched trace gives an invoke the rows its input "
                    "has — the other invokes would read the wrong ones. Trace a "
                    "chunked input on its own."
                )
            items.extend(rows)
            forward.update(forward_kwargs)

        encoding = self._collate(items)
        self._supply_position_ids(encoding)
        return tuple(), {**encoding, **forward}

    def _preprocess_invoke(self, data: Any, kwargs: dict) -> tuple:
        """One invoke -> (list of per-row model-input dicts, forward kwargs).

        Returns ``(None, kwargs)`` for an opaque input (a raw feature tensor or a
        multimodal encoding) that the caller passes through to the model untouched.
        """
        media = self._as_processor_encoding(data, kwargs)
        if media is not None:
            return None, media
        if self._is_opaque(data, kwargs):
            return None, kwargs
        if self.task == "keypoint-matching":
            # This task's unit input is a *pair* of images, which collides with
            # the list convention (one prompt per element): the pair is split
            # into two single-image preprocess calls, and a nested pair reads
            # as pre-tokenized ids — which is why this check sits before
            # `_is_pretokenized`. (An encoding you built yourself is opaque and
            # never reaches here.)
            raise NotImplementedError(
                "task='keypoint-matching' takes a pair of images as one input, "
                "which a trace's list convention (one prompt per element) "
                "would split. Run the whole task with model.pipe([image_a, "
                "image_b]), or trace one forward on an encoding you build "
                "yourself: model.image_processor(images=[image_a, image_b], "
                "return_tensors='pt')."
            )
        if self._is_pretokenized(data, kwargs):
            return self._encode_pretokenized(data, kwargs)
        if self.task == "mask-generation":
            # This task's preprocess *runs the model*: it embeds the image, then
            # yields one input per batch of candidate points, each carrying a copy
            # of that embedding. There is no single forward to assemble — the
            # encoder ran outside the trace, and the rows would be one copy of the
            # image embedding per point batch (128 of them at the task's default).
            raise NotImplementedError(
                "task='mask-generation' has no forward to trace from an image: its "
                "preprocess embeds the image by running the model, then yields one "
                "input per batch of candidate points. Run the whole task with "
                "model.pipe(image), or trace one forward on an encoding you build "
                "yourself: model.image_processor(image, return_tensors='pt'), with "
                "the points you want as input_points=."
            )
        # Text / image / audio: let the pipeline tokenize/featurize it, routing the
        # invoke's kwargs (truncation, chat tools, ...) through its own param split.
        preprocess_params, forward_params, _ = self.pipeline._sanitize_parameters(**kwargs)
        # Chat message(s) are wrapped in Chat (as Pipeline.__call__ would) so the
        # template is applied; otherwise a list of strings is one input per prompt.
        inputs = self._as_chats(data, self._chat_cls())
        if inputs is None:
            inputs = list(data) if isinstance(data, (list, tuple)) else [data]
            inputs = self._parse_task_args(inputs)
        rows = []
        for one in inputs:
            row = self.pipeline.preprocess(one, **preprocess_params)
            # A chunked task's preprocess is a generator: it *yields* the
            # encodings it splits one input into instead of returning one, and
            # each is a forward of its own. They are unrolled into rows here, so
            # the whole input is traced in the trace's one forward; handing the
            # generator to `_collate` is what makes it ask a generator for
            # `.items()`.
            rows.extend([row] if hasattr(row, "items") else row)
        merged = [self._merge_nested_encodings(row) for row in rows]
        if any(row is not None for row in merged):
            # A dual-encoder zero-shot task (CLIP, CLAP) runs one forward whose
            # batch dims differ per half — one image/audio row against one text
            # row per candidate label — so its rows don't collate with anything
            # else's; a lone one goes to the model whole, like an encoding.
            if len(rows) > 1:
                raise NotImplementedError(
                    f"task={self.task!r} pairs each input with its own nested "
                    "text encoding, so several inputs don't collate into one "
                    "forward. Trace one input at a time."
                )
            encoding = {
                key: value
                for key, value in merged[0].items()
                if isinstance(value, torch.Tensor)
            }
            return None, {**encoding, **forward_params}
        return rows, forward_params

    def _parse_task_args(self, inputs: list) -> list:
        """Run the pipeline's ``_args_parser`` over task-input dicts.

        Some input normalization lives in the parser ``Pipeline.__call__``
        invokes, not in ``preprocess``: ``table-question-answering`` turns the
        task dict's ``table`` into the ``pd.DataFrame`` its preprocess requires
        there. Calling ``preprocess`` directly would skip it.
        """
        parser = getattr(self.pipeline, "_args_parser", None)
        if parser is None:
            return inputs
        parsed = []
        for one in inputs:
            if hasattr(one, "keys") and self._is_task_input(one):
                out = parser(one)
                parsed.extend(out if isinstance(out, list) else [out])
            else:
                parsed.append(one)
        return parsed

    @staticmethod
    def _merge_nested_encodings(row: Any) -> Optional[dict]:
        """Flatten a preprocess row whose model inputs sit one level down.

        A dual-encoder zero-shot pipeline (CLIP, CLAP) returns the candidate
        labels' text encoding *nested* — ``{"pixel_values": ..., "text_inputs":
        [BatchEncoding]}`` — and unwraps it in its ``_forward`` right before
        the model call. Collation keeps only top-level tensors, which would
        silently drop the text half. Returns the row with every nested
        encoding's tensors merged in, or ``None`` when nothing is nested.
        """
        merged, found = {}, False
        for key, value in row.items():
            inner = value
            if isinstance(inner, (list, tuple)) and len(inner) == 1:
                inner = inner[0]
            if hasattr(inner, "keys") and not isinstance(inner, torch.Tensor):
                inner = dict(inner)
                if inner and all(
                    isinstance(item, torch.Tensor) for item in inner.values()
                ):
                    merged.update(inner)
                    found = True
                    continue
            merged[key] = value
        return merged if found else None

    def _as_processor_encoding(self, data: Any, kwargs: dict) -> Optional[dict]:
        """Run the task's processor when an invoke is written in processor terms.

        ``trace(prompt, images=[img])`` and ``trace(text=prompt, images=[img])`` name
        the *processor's* arguments, not the model's. Without this they are handed to
        the model untouched, which raises from deep inside modeling code
        (``You must specify exactly one of input_ids or inputs_embeds``) — an error
        that says nothing about the real problem. ``generate`` has always run the
        processor for these; this makes ``trace``/``scan`` agree with it.

        Returns the model-input encoding merged with any leftover forward kwargs, or
        ``None`` when this isn't a processor call and the usual routing should apply.
        """
        if self.processor is None or not (set(kwargs) & self._PROCESSOR_MEDIA_KEYS):
            return None

        call = {key: value for key, value in kwargs.items() if key in self._PROCESSOR_KEYS}
        forward = {
            key: value for key, value in kwargs.items() if key not in self._PROCESSOR_KEYS
        }

        if data is not None:
            if "text" in call:
                raise ValueError(
                    "Got the prompt both positionally and as `text=`; pass just one."
                )
            call["text"] = data

        # Featurizing an image goes through numpy, which a fake-tensor mode refuses.
        # `scan` runs the whole batch step under one, so step outside it here: the
        # encoding is cheap, real, and `allow_non_fake_inputs` lets it into the
        # faked forward.
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            encoding = self.processor(**call, return_tensors="pt")
        return {**dict(encoding), **forward}

    def _collate(self, items: list) -> dict:
        """Pad per-invoke encodings into one batch of model-input tensors."""
        # Drop the pipeline's non-tensor bookkeeping (e.g. prompt_text) up front, so
        # every item has the same keys for pad_collate_fn's consistency check.
        items = [
            {k: v for k, v in item.items() if isinstance(v, torch.Tensor)}
            for item in items
        ]
        if len(items) == 1:
            return dict(items[0])
        from transformers.pipelines.base import pad_collate_fn

        feature = self.feature_extractor or self.image_processor
        return dict(pad_collate_fn(self.tokenizer, feature)(items))

    @staticmethod
    def _is_opaque(data: Any, kwargs: dict) -> bool:
        """Whether the input must be passed to the model as-is (not batched as text).

        A raw feature tensor, or an encoding (positional or via ``kwargs``) that has
        no ``input_ids`` or carries a non-text modality field (``pixel_values``, ...).
        """
        if isinstance(data, torch.Tensor):
            return data.is_floating_point()
        if data is None:
            return TransformersModel._has_nontext_keys(kwargs)
        if hasattr(data, "get") and not isinstance(data, (list, tuple, str)):
            if TransformersModel._is_task_input(data):
                return False
            return data.get("input_ids") is None or TransformersModel._has_nontext_keys(data)
        return False

    @staticmethod
    def _is_task_input(data: Any) -> bool:
        """Whether a mapping is the *task's* own input rather than model inputs.

        Some tasks take a dict — ``{"image": ..., "question": ...}`` for
        ``document-question-answering``, ``{"image": ..., "candidate_labels":
        [...]}`` for ``zero-shot-object-detection`` — which is what their
        ``preprocess`` turns into model inputs. Passed to the model as an encoding
        it fails deep in modeling code (``missing 2 required positional
        arguments``), naming nothing the caller wrote.

        Model inputs are tensors, so a mapping holding none of them is not an
        encoding: that, rather than a list of task names, is what tells the two
        apart. And it must be *tensors* specifically — a shape-duck-typed check
        misreads ``table-question-answering``'s dict, whose ``pd.DataFrame``
        table also has a ``.shape``, as an encoding.
        """
        values = list(dict(data).values()) if hasattr(data, "keys") else []
        return bool(values) and not any(
            isinstance(value, torch.Tensor) for value in values
        )

    @staticmethod
    def _has_nontext_keys(encoding: Any) -> bool:
        """Whether an encoding carries a field beyond the plain text/token ones."""
        return any(
            key not in TransformersModel._TEXT_KEYS and value is not None
            for key, value in dict(encoding).items()
        )

    @staticmethod
    def _is_pretokenized(data: Any, kwargs: dict) -> bool:
        """Whether the input is already token ids / an encoding (not raw text)."""
        if data is None:
            return kwargs.get("input_ids") is not None
        if isinstance(data, str):
            return False
        if isinstance(data, torch.Tensor):
            return not data.is_floating_point()
        if isinstance(data, (list, tuple)):
            if not data:
                return False
            first = data[0]
            if isinstance(first, str):
                return False
            if isinstance(first, torch.Tensor):
                return not first.is_floating_point()
            return isinstance(first, (int, list, tuple))
        if hasattr(data, "get"):
            return data.get("input_ids") is not None
        return False

    def _encode_pretokenized(self, data: Any, kwargs: dict) -> tuple:
        """A pre-tokenized invoke -> (per-row ``{input_ids, attention_mask}`` items,
        forward kwargs).

        Only plain token ids reach here — a multimodal encoding is opaque and passes
        through untouched — so every row is split to one ``[1, L]`` item for padding.
        """
        if data is None:
            ids, masks = kwargs.get("input_ids"), kwargs.get("attention_mask")
            forward = {k: v for k, v in kwargs.items() if k not in self._TEXT_KEYS}
        elif hasattr(data, "get") and not isinstance(data, (list, tuple, torch.Tensor)):
            ids, masks = data["input_ids"], data.get("attention_mask")
            forward = dict(kwargs)
        else:
            ids, masks = data, None
            forward = dict(kwargs)

        id_rows = self._as_sequences(ids)
        mask_rows = self._as_sequences(masks) if masks is not None else None
        items = []
        for index, sequence in enumerate(id_rows):
            input_ids = torch.tensor(sequence).unsqueeze(0)
            mask = (
                torch.tensor(mask_rows[index]).unsqueeze(0)
                if mask_rows is not None
                else torch.ones_like(input_ids)
            )
            items.append({"input_ids": input_ids, "attention_mask": mask})
        return items, forward

    @staticmethod
    def _as_sequences(value: Any) -> list:
        """Split ids/masks into a list of 1-D python-int sequences (one per row)."""
        if isinstance(value, torch.Tensor):
            value = value.tolist()  # 1-D -> list[int]; 2-D -> list[list[int]]
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            first = value[0]
            if isinstance(first, (list, tuple)):
                return [list(sequence) for sequence in value]
            if isinstance(first, torch.Tensor):
                return [sequence.tolist() for sequence in value]
            # A flat list of ints is a single sequence.
            return [list(value)]
        return [list(value)]

    def _supply_position_ids(self, encoding: dict) -> None:
        """Add mask-derived ``position_ids`` for a left-padded text batch (in place).

        Left padding shifts each real token's absolute index, so an absolute-position
        model (GPT-2 family) would mispredict a short prompt padded up to a longer
        one. Deriving ``position_ids`` from the attention mask keeps every real token
        at its true 0-based position. Only applied to a genuinely left-padded,
        text-only batch: an *unpadded* batch needs no correction (caught by
        ``mask.all()``), a right-padded (encoder) batch is already correct, and a
        multimodal model derives its own positions from the image-expanded sequence.
        Row count is deliberately not part of this test -- a single padded row needs
        the correction just as much as a padded batch does, and gating on
        ``shape[0] > 1`` made the same prompt answer differently depending on whether
        another row happened to share its batch.
        """
        mask = encoding.get("attention_mask")
        # Under `scan` the forward runs on fake tensors to propagate shapes only, so
        # there are no real mask values to read -- `bool(mask.all())` would raise
        # GuardOnDataDependentSymNode. position_ids do not affect shapes, so skipping
        # the correction here changes nothing a scan can observe.
        if detect_fake_mode() is not None:
            return
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dim() != 2
            or bool(mask.all())
            or getattr(self.tokenizer, "padding_side", None) != "left"
            or any(key not in self._TEXT_KEYS for key in encoding)
        ):
            return
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 0)
        encoding["position_ids"] = position_ids
