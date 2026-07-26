"""Trace interventions on a vLLM inference engine.

vLLM runs the model in its own worker process, so a trace cannot simply run
alongside it the way it does for a local ``nn.Module``: this process holds only a
meta-device copy of the module tree, with no weights to hook. The intervention
therefore travels *to* the model. Each invoke's worker is serialized into its
request's ``SamplingParams.extra_args`` and rides vLLM's own request pipeline into
the worker, where [`GPUModelRunner`][nnsight.modeling.vllm.model_runners.GPUModelRunner]
deserializes it, runs it against the real module, and ships saved values back.

Two consequences shape everything here. Interventions are scoped to a *request*,
so each invoke carries exactly one prompt — batching several prompts means several
``tracer.invoke(...)`` blocks, not a list. And because the engine decides when a
request runs, an activation arrives as a flat ``[total_tokens, hidden]`` slab of
whatever the scheduler packed into that step rather than a padded ``[batch, seq]``
stack; [`VLLMBatcher`][nnsight.modeling.vllm.batching.VLLMBatcher] is what maps a worker
onto its own tokens within it.
"""

from __future__ import annotations

import atexit
import contextlib
from typing import TYPE_CHECKING, Any, Callable

import torch

from ...intervention.eproperty import eproperty
from ...intervention.serialization import dumps
from ..mixins.remotable import Remotable

if TYPE_CHECKING:
    from torch.nn import Module


class VLLM(Remotable):
    """A vLLM engine whose internals can be traced.

    Interventions are written exactly as for any other model — the module tree
    mirrors the architecture vLLM loaded — but they run inside the engine's worker
    process. Sampling settings (``temperature``, ``max_tokens``, ``top_p``, ...)
    are passed to ``trace``/``invoke`` rather than configured on the model, since
    each invoke is its own vLLM request. Read generated tokens through
    ``model.logits`` / ``model.samples`` (or the streamed output in async), not
    ``tracer.result`` — the latter is not served here.

    Examples:
        Single prompt, edit an activation, read the logits::

            >>> model = VLLM("gpt2", dispatch=True)
            >>> with model.trace("The Eiffel Tower is in", temperature=0.0):
            ...     model.transformer.h[8].output[:] = 0
            ...     logits = model.logits.save()
            >>> model.tokenizer.decode(logits.argmax(dim=-1))

        Several prompts is several ``invoke`` blocks (each is one request), not a
        list — a shared save escapes each into its own name::

            >>> with model.trace(temperature=0.0) as tracer:
            ...     with tracer.invoke("The Eiffel Tower is in"):
            ...         a = model.logits.save()
            ...     with tracer.invoke("The capital of Japan is"):
            ...         b = model.logits.save()

        Streaming with ``mode="async"`` — saves arrive on the finished output::

            >>> model = VLLM("gpt2", dispatch=True, mode="async")
            >>> with model.trace("Hello", max_tokens=5) as tracer:
            ...     logits = model.logits.save()
            >>> async for output in tracer.backend:  # doctest: +SKIP
            ...     last = output
            >>> last.saves["logits"]                 # doctest: +SKIP

        A GPU-less client running a trace on a remote nnsight-serve engine — the
        client only builds a meta tree and never dispatches::

            >>> model = VLLM("gpt2")                                  # no GPU needed
            >>> with model.trace("Hello", serve="http://host:8000"):  # doctest: +SKIP
            ...     logits = model.logits.save()

    Attributes:
        vllm_entrypoint: The underlying ``vllm.LLM``, or None until dispatch.
        tokenizer: The tokenizer vLLM resolved for the checkpoint.
    """

    def __init__(self, *args: Any, mode: str = "sync", **kwargs: Any) -> None:
        self.vllm_entrypoint = None
        self.tokenizer = None
        # ``mode="async"`` builds vLLM's streaming ``AsyncLLM`` instead of the
        # synchronous ``LLM``; a trace then yields its outputs as they generate
        # through ``async for output in tracer.backend``.
        self._async_engine = mode == "async"
        # Whether this construction brought up the process group — so only then do we
        # tear it down (on dispatch and at exit), never a group nnsight found running.
        self._owns_distributed = False

        # Model-parallel init has to happen before `Meta.__init__` opens its
        # meta-device context: vLLM builds real rank tensors here and later calls
        # `.tolist()` on them, which a meta tensor cannot serve.
        if not torch.distributed.is_initialized():
            self._init_distributed()
            self._owns_distributed = True
            # Tear down only the group nnsight brought up — not one already running,
            # and not once per construction (atexit does not dedupe).
            atexit.register(VLLM._cleanup_distributed)

        super().__init__(*args, **kwargs)

    @staticmethod
    def _init_distributed() -> None:
        """Bring up a single-rank gloo process group on a free local port."""
        import socket

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0, backend="gloo")
        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

    @staticmethod
    def _cleanup_distributed() -> None:
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        for teardown in (destroy_model_parallel, destroy_distributed_environment):
            try:
                teardown()
            except Exception:
                pass

    @eproperty(description="pre-sampling logits for this step")
    def logits(self, value: Any) -> Any:
        """The logits for this request's step, before sampling.

        A hookable run-level value like a module's ``.output`` — reading it parks the
        worker until the engine produces this step's logits; writing it swaps them.
        Under ``tracer.iter`` each pass sees the next decoded step's logits::

            with model.trace("Hello", temperature=0.0) as tracer:
                logits = model.logits.save()
        """
        return value

    @eproperty(description="token ids drawn from logits this step")
    def samples(self, value: Any) -> Any:
        """The token ids the sampler drew from [`logits`][nnsight.modeling.vllm.vllm.VLLM.logits] for this step.

        Read or edit them inside a trace; setting them replaces the tokens the engine
        continues generation from — force a token::

            with model.trace("Hello", temperature=0.0, max_tokens=3) as tracer:
                for _ in tracer.iter[:3]:
                    model.samples = torch.zeros_like(model.samples)  # feed token 0
        """
        return value

    @staticmethod
    @contextlib.contextmanager
    def _meta_device():
        """Let the meta tree build on a node with no GPU.

        Constructing the tree makes vLLM pick an attention backend, which probes
        the GPU's compute capability to choose a flash-attention version. The tree
        is only ever read for its structure — a client never runs a forward — so on
        a GPU-less node that probe is answered with a stand-in. A node with a real
        GPU (a server) answers it itself, so nothing is faked there. A truly
        CPU-only host (no CUDA at all) selects a CPU backend that never probes, so
        this is a no-op there too.
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            yield
            return
        from unittest import mock

        with mock.patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            yield

    def _load_meta(self, repo_id: str, **kwargs: Any) -> "Module":
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
        from vllm.tokenizers import cached_tokenizer_from_config

        # The meta tree only needs the architecture, so build it single-rank
        # regardless of the parallelism the real engine will use.
        kwargs = {**kwargs, "tensor_parallel_size": 1, "pipeline_parallel_size": 1}

        with self._meta_device():
            vllm_config = EngineArgs(model=repo_id, **kwargs).create_engine_config()
            vllm_config.load_config.device = "meta"

            with set_current_vllm_config(vllm_config):
                loader = DummyModelLoader(vllm_config.load_config)
                # DummyModelLoader still fills its dummy weights; the tree is only
                # needed for its structure, so skip the fill entirely.
                loader.load_weights = lambda *args, **kwargs: None
                model = loader.load_model(vllm_config, vllm_config.model_config)

        # Rotary embeddings are cached globally by config, so a meta-built entry
        # would be handed to the real engine on dispatch.
        _ROPE_DICT.clear()

        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return model

    # vLLM already carries interventions through to the worker in this field, so
    # the worker needs no transport of its own.
    _WORKER_CLS = "nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"

    # A ready module (the worker-side runner wrapping the module vLLM already
    # loaded) takes the base `_wrap` path: nothing to build — no engine, no
    # meta tree; the caller sets the tokenizer.

    def _load(self, repo_id: str, **kwargs: Any) -> "Module":
        meta_model = self._load_meta(repo_id, **kwargs)

        # The real engine brings up its own process group; the one __init__ made
        # to build the meta tree would collide with it. Only tear down a group this
        # construction created — never one the caller already had running.
        if self._owns_distributed:
            self._cleanup_distributed()

        if self._async_engine:
            self.vllm_entrypoint = self._load_async(repo_id, **kwargs)
        else:
            self.vllm_entrypoint = self._load_sync(repo_id, **kwargs)

        return meta_model

    def _load_sync(self, repo_id: str, **kwargs: Any) -> Any:
        from vllm import LLM

        from .engines.engine import NNsightLLMEngine

        llm = LLM(
            repo_id,
            worker_cls=self._WORKER_CLS,
            # Hooks cannot fire inside a captured CUDA graph, which freezes the
            # ops it replays.
            enforce_eager=True,
            **kwargs,
        )
        # step() collects each finished request's saves; see NNsightLLMEngine.
        llm.llm_engine.__class__ = NNsightLLMEngine
        return llm

    def _load_async(self, repo_id: str, **kwargs: Any) -> Any:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        # AsyncLLM runs its own output-handler loop rather than a synchronous
        # step(), so saves are collected by the streaming backend instead (see
        # nnsight.modeling.vllm.async_backend), and no engine subclass is needed.
        engine_args = AsyncEngineArgs(
            model=repo_id,
            worker_cls=self._WORKER_CLS,
            enforce_eager=True,
            **kwargs,
        )
        return AsyncLLM.from_engine_args(engine_args)

    def _batch_size(self, *inputs: Any, **kwargs: Any) -> int:
        """Number of batch rows an invoke contributes — one request, so one row.

        Keyword arguments are sampling settings rather than data, so an invoke with
        only kwargs contributes nothing and sees the whole batch.
        """
        return 1 if inputs else 0

    def _batch(self, invokes: list[tuple], fn: Any) -> tuple:
        """Turn each invoke into one vLLM request.

        Unlike a stacked-tensor model there is nothing to pad or combine: the
        engine batches requests itself, so this only converts each invoke's input
        into a prompt and its kwargs into that request's ``SamplingParams``.

        Returns:
            ``((prompts, params, lora_requests), {})`` for the traced call.
        """
        from vllm import SamplingParams

        prompts, params, lora_requests = [], [], []
        for inputs, kwargs in invokes:
            kwargs = dict(kwargs)
            lora_requests.append(kwargs.pop("lora_request", None))
            prompts.append(self._prompt(*inputs))
            params.append(SamplingParams(**kwargs))

        return (prompts, params, lora_requests), {}

    def _prompt(self, *inputs: Any) -> Any:
        """Convert one invoke's input into a vLLM prompt.

        Accepts a string, a list of token ids, or a tokenizer's output dict. A
        request is one sequence, so anything carrying several prompts is rejected
        here rather than silently generating from the first.
        """
        from vllm.inputs import TokensPrompt

        if len(inputs) != 1:
            raise ValueError(
                f"Each invoke takes exactly one prompt, got {len(inputs)}. "
                "Use a separate tracer.invoke(...) per prompt."
            )
        prompt = inputs[0]

        if isinstance(prompt, dict):
            return self._tokenized_prompt(prompt)
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, (list, tuple)):
            if not prompt:
                raise ValueError("Empty prompt")
            if isinstance(prompt[0], int):
                return TokensPrompt(prompt_token_ids=list(prompt))
            raise ValueError(
                "Multiple prompts per invoke are not supported. "
                "Use a separate tracer.invoke(...) per prompt."
            )
        return prompt

    def _tokenized_prompt(self, inputs: dict) -> Any:
        """Convert a tokenizer's ``{input_ids, attention_mask}`` output to a prompt.

        vLLM has no padding to mask — a request is exactly its own tokens — so a
        mask, if given, selects which ids survive.
        """
        from vllm.inputs import TokensPrompt

        input_ids = inputs["input_ids"]
        mask = inputs.get("attention_mask")

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(mask, torch.Tensor):
            mask = mask.tolist()

        if not input_ids:
            raise ValueError("Empty prompt")
        # A tokenizer emits [[ids]] for one prompt and [ids] when unbatched.
        if isinstance(input_ids[0], list):
            if len(input_ids) > 1:
                raise ValueError(
                    "Multiple prompts per invoke are not supported. "
                    "Use a separate tracer.invoke(...) per prompt."
                )
            input_ids = input_ids[0]
            mask = mask[0] if mask else None

        if mask is not None:
            input_ids = [i for i, m in zip(input_ids, mask) if m != 0]

        return TokensPrompt(prompt_token_ids=input_ids)

    def trace(self, *inputs: Any, **kwargs: Any) -> Any:
        from .tracer import VLLMTracer

        # vLLM generation length is `max_tokens`; accept `max_new_tokens` (from the
        # LanguageModel API) on trace and generate alike, rewriting before it reaches
        # SamplingParams (where an unknown kwarg now raises).
        if "max_new_tokens" in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")

        # `serve=url` runs the trace on a remote nnsight-serve engine; `api_key`
        # rides along with it. Pop both before they reach the base trace.
        serve = kwargs.pop("serve", None)
        api_key = kwargs.pop("api_key", None)

        # A tracer whose worker-building the async/serve backends can call without
        # running a forward (VLLMTracer.prepare). Keeps the base tracer untouched.
        kwargs.setdefault("tracer_cls", VLLMTracer)
        if serve is not None and kwargs.get("backend") is None:
            from .serve.backend import LocalServeBackend

            kwargs["backend"] = LocalServeBackend(self, serve, api_key=api_key)
        # On an async engine the trace streams its outputs; the backend submits the
        # request and yields them, in place of running the forward here.
        elif (
            self._async_engine
            and kwargs.get("backend") is None
            and not kwargs.get("remote")
        ):
            from .async_backend import AsyncVLLMBackend

            kwargs["backend"] = AsyncVLLMBackend(self)
        # The traced call is the engine request, not the meta module's forward:
        # the module here has no weights to run.
        kwargs.setdefault("fn", self._call)
        return super().trace(*inputs, **kwargs)

    def generate(self, *inputs: Any, **kwargs: Any) -> Any:
        """Alias for `trace`, for parity with other models' ``generate``.

        vLLM generation is driven by ``max_tokens`` (``trace`` rewrites
        ``max_new_tokens`` to it), so there is no forward/generate distinction to draw.
        Read generated tokens through ``model.logits``/``model.samples`` under
        ``tracer.iter``, not ``tracer.result`` — the latter is never served here and a
        worker reading it would park forever.
        """
        return self.trace(*inputs, **kwargs)

    def _call(
        self, prompts: list, params: list, lora_requests: list, **kwargs: Any
    ) -> Any:
        """Run the engine with this trace's workers attached to its requests."""
        mediators = self._attach_mediators(params, **kwargs)
        outputs = self.vllm_entrypoint.generate(
            prompts, sampling_params=params, lora_request=lora_requests
        )
        self._collect(mediators, outputs)
        return outputs

    @staticmethod
    def _collect(mediators: list, outputs: list) -> None:
        """Bring each request's saved values home to the worker that asked for them.

        The workers ran in another process, so the values here are new objects: mark
        them saved in *this* process, and write them into the worker's scope, which
        is where the tracer reads a block's results from once the run is over. A
        worker that raised carries its error back too; re-raise the first real one
        (a ``tracer.stop()`` is control flow and stays silent).
        """
        from ...intervention.errors import raise_deferred
        from ...tracing.tracer import mark
        from .collect import merge_shared_saves

        per_request_saves = []
        for mediator, output in zip(mediators, outputs):
            saves = getattr(output, "saves", {})
            per_request_saves.append(saves)
            for name, value in saves.items():
                mark(value)  # marking results after the run; no trace active to guard
                mediator.lcls[name] = value

        # A name saved above the invoke blocks ships back once per request,
        # each copy carrying that request's writes. Merge the copies
        # element-wise so the result reads as if every invoke had mutated one
        # shared object (the local semantics). The merged containers are new
        # objects; mark them saved so the result push keeps them.
        for value in merge_shared_saves(mediators, per_request_saves).values():
            mark(value)

        for output in outputs:
            raise_deferred(getattr(output, "nnsight_error", None))

    def _attach_mediators(self, params: list, **kwargs: Any) -> list:
        """Serialize each invoke's worker into its request's ``extra_args``.

        ``extra_args`` is a stock ``SamplingParams`` field that vLLM already carries
        through to the worker, so the worker needs no transport of its own. Sampling
        settings given to ``trace`` itself fill in for any request that did not set
        them on its own invoke.

        A worker with no batch group is an invoke with no prompt — it has no vLLM
        request to ride, since each invoke *is* one request. An empty ``tracer.invoke()``
        with a do-nothing body is a harmless no-op and is dropped, but one carrying
        interventions would vanish silently, so that is refused. Unknown ``trace``
        keyword arguments (a typo'd sampling setting) are refused too, rather than
        silently ignored the way the fill loop otherwise would.

        Returns:
            The workers that were attached, in request order.
        """
        from vllm import SamplingParams

        from ...tracing.tracer import skippable

        attached, orphaned = [], []
        for mediator in self.interleaver.mediators:
            (attached if mediator.batch_group is not None else orphaned).append(mediator)

        for mediator in orphaned:
            if mediator.node is not None and skippable(mediator.node):
                raise ValueError(
                    "A `tracer.invoke(...)` with no prompt has no vLLM request to run "
                    "on, so its interventions would be silently dropped. Each invoke is "
                    "one request — give every invoke a prompt, or remove the empty invoke."
                )

        default = SamplingParams()
        for attr in kwargs:
            if not hasattr(default, attr):
                raise TypeError(
                    f"unexpected trace argument {attr!r}: not a vLLM SamplingParams "
                    "field. trace()/invoke() keyword arguments are sampling settings "
                    "(temperature, top_p, max_tokens, ...)."
                )

        for mediator, param in zip(attached, params):
            param.extra_args = {"nnsight_mediator": dumps(mediator)}
            for attr, value in kwargs.items():
                if getattr(param, attr) == getattr(default, attr):
                    setattr(param, attr, value)

        return attached

    def interleave(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Dispatch the trace to the engine instead of running it here.

        Overrides [`interleave`][nnsight.intervention.envoy.Envoy.interleave], which starts
        each worker in this process alongside the model's forward. There is no
        forward to run here — the weights live in the engine's worker — so the
        workers are not started; they are serialized onto the requests by
        `_call` and started by the model runner on the other side.
        """
        if not self.dispatched:
            self.dispatch()
        # The caller (VLLMTracer.execute) cancels the interleaver in its own finally,
        # covering both this normal return and a failure before it — so no cancel here.
        return fn(*args, **kwargs)

    def _remoteable_model_key(self) -> str:
        return self.args[0]

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs: Any) -> "VLLM":
        return cls(model_key, **kwargs)

    def _remoteable_persistent_objects(self) -> dict:
        objects = super()._remoteable_persistent_objects()
        objects["Tokenizer"] = self.tokenizer
        return objects

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        # The engine is a live process handle; the far side has its own.
        state["vllm_entrypoint"] = None
        state["_async_engine"] = self._async_engine
        if self.tokenizer is not None:
            self.tokenizer._persistent_id = "Tokenizer"
        state["tokenizer"] = self.tokenizer
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self.vllm_entrypoint = state["vllm_entrypoint"]
        self._async_engine = state.get("_async_engine", False)
        self.tokenizer = state["tokenizer"]
