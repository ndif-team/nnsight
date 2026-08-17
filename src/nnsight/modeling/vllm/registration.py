"""Interventions that stay on the engine and run for every request.

A trace ships its block to the worker on the request it rides, so a sweep of a
thousand prompts serializes and deserializes the same block a thousand times, and
only requests that *are* nnsight traces are touched at all.

A registration inverts that. The block goes to the worker once and stays there;
from then on every request the engine runs gets its own copy of it — including
requests submitted by something that has never heard of nnsight, an OpenAI-API
client on the same server. Each copy keeps its own scope, so what it saves is
that request's own, and the values accumulate on the worker until they are
collected in one trip.

Example:
    >>> model = VLLM("meta-llama/Llama-3.1-8B", dispatch=True)
    >>>
    >>> with model.register() as (tracer, registration):   # doctest: +SKIP
    ...     hidden = model.model.layers[16].output[0].save()
    >>>
    >>> model.generate("The Eiffel Tower is in", max_tokens=5)   # doctest: +SKIP
    >>> model.generate("The capital of Japan is", max_tokens=5)  # doctest: +SKIP
    >>>
    >>> {request: saved["hidden"].shape                # doctest: +SKIP
    ...  for request, saved in registration.saves.items()}
    >>> registration.clear()                        # doctest: +SKIP

The block is written exactly like a trace body — the same envoy tree, the same
``.save()``. What it cannot do is anything that belongs to one particular
request: there is no prompt to invoke, so ``tracer.invoke(...)`` has no meaning
here, and the block applies to whatever the engine happens to run.
"""

from __future__ import annotations

import pickle
import uuid
import warnings
from types import CodeType
from typing import TYPE_CHECKING, Any

from ...intervention.interleaver import Mediator
from ...intervention.serialization import dumps
from ...intervention.tracer import InterleavingTracer
from ...tracing.backend import Backend
from ...tracing.tracer import skip_context, skippable

if TYPE_CHECKING:
    from .vllm import VLLM


def _warn_if_prefix_caching(model: "VLLM") -> None:
    """Say so if the engine will serve some tokens without a forward pass.

    A prefix-cached token never runs, so no hook fires for it and a registered
    block sees a short activation — quietly, with nothing to indicate it. A trace
    can avoid this by asking for its own request to be recomputed (see
    ``_attach_mediators``), but a registration rides requests it did not create
    and often did not even come from nnsight, so there is nothing to set the flag
    on. The engine has to be built with ``enable_prefix_caching=False`` instead,
    and it is worth saying at the moment the mistake is made rather than leaving
    it to be found in the shapes.
    """
    engine = model.vllm_entrypoint
    if engine is None:
        return
    try:
        core = getattr(engine, "llm_engine", engine)
        cache_config = core.vllm_config.cache_config
    except AttributeError:
        return
    if getattr(cache_config, "enable_prefix_caching", False):
        warnings.warn(
            "This engine has prefix caching on, so tokens served from the cache "
            "never run a forward pass and a registered block will not see them — "
            "its activations come back short, with no error. Build the model with "
            "VLLM(..., enable_prefix_caching=False) to register against whole "
            "prompts.",
            stacklevel=4,
        )


class Registration:
    """A handle on a block the engine is running for every request.

    Returned by [`VLLM.register`][nnsight.modeling.vllm.vllm.VLLM.register] and
    live until `clear`. Values pile up on the worker as requests finish;
    [`saves`][nnsight.modeling.vllm.registration.Registration.saves] reads them
    and `drain` takes them, neither of which stops the registration — so a long
    sweep can be emptied in batches while it keeps running.

    Attributes:
        model: The engine this is registered on.
        id: The engine-wide name for this registration, used to address it in the
            worker and to key its side of a collect.
    """

    def __init__(self, model: "VLLM", id: str) -> None:
        self.model = model
        self.id = id
        self.cleared = False

    def _rpc(self, method: str, *args: Any) -> list:
        engine = self.model.vllm_entrypoint
        # The sync entrypoint keeps the engine one level down; the async one is
        # already the engine.
        core = getattr(engine, "llm_engine", engine)
        return core.collective_rpc(method, args=args)

    @property
    def saves(self) -> dict[str, dict[str, Any]]:
        """What each finished request's copy of the block saved, by request.

        ``{request_id: {name: value}}``, named the same way a trace's values come
        back (``output.saves["logits"]``) because they are the same thing — the
        block's ``.save()``\\ d locals — just one set per request rather than one
        set in total. The request id is the engine's own, so it matches the
        ``RequestOutput`` that produced it.

        Reading this does not drop anything: a value stays until `drain` or
        `clear` takes it, so reading twice gives the same answer. Over a long
        sweep that accumulates on the worker — use `drain` there.
        """
        return self._fetch(clear=False)

    def drain(self) -> dict[str, dict[str, Any]]:
        """`saves`, and drop what is returned from the worker.

        What a sweep wants: take this batch's values home and leave nothing
        behind, so the next read is the next batch and the worker's memory does
        not grow with the number of requests served.
        """
        return self._fetch(clear=True)

    def _fetch(self, clear: bool) -> dict[str, dict[str, Any]]:
        from ...intervention.errors import raise_deferred

        if self.cleared:
            raise ValueError(
                "This registration was cleared; register again to collect more."
            )
        payloads = self._rpc("nnsight_collect_registered", self.id, clear)
        collected: dict[str, dict[str, Any]] = {}
        deferred = None
        # Under pipeline parallelism each stage holds the part of the block that
        # ran on it, so merge rather than take the first non-empty answer.
        for payload in payloads or ():
            if payload is None:
                continue
            for request_id, entry in pickle.loads(payload).items():
                collected.setdefault(request_id, {}).update(entry["saves"])
                if deferred is None:
                    deferred = entry["error"]
        # The block runs far from whoever wrote it, so an error in it has no
        # other way home. Raised here rather than returned, so a registration
        # that is quietly failing on every request cannot be mistaken for one
        # that is merely finding nothing.
        raise_deferred(deferred)
        return collected

    def clear(self) -> None:
        """Stop running the block and drop anything it has not handed back."""
        if self.cleared:
            return
        self._rpc("nnsight_clear_registered", self.id)
        self.cleared = True

    def __repr__(self) -> str:
        state = "cleared" if self.cleared else "active"
        return f"<Registration {self.id} ({state})>"


class RegisteringTracer(InterleavingTracer):
    """Capture a block and leave it on the engine instead of running it once.

    The counterpart of [`EditingTracer`][nnsight.intervention.editing.EditingTracer]
    for a runtime whose model lives in another process: an edit is replayed by
    the envoy that stores it, which on vLLM would leave it in the client, where
    there are no weights. This sends the block across instead, and hands back a
    [`Registration`][nnsight.modeling.vllm.registration.Registration] to collect
    through.
    """

    def __init__(self, model: "VLLM", *, backend: Backend | None = None) -> None:
        super().__init__(model, "__call__", backend=backend)
        self._model = model
        self.registration: Registration | None = None

    def __enter__(self) -> tuple["RegisteringTracer", Registration]:
        """Enter the block, binding the tracer and the handle results come back through.

        Both, the way [`Envoy.edit`][nnsight.intervention.envoy.Envoy.edit] binds
        ``(tracer, edited)``: a registered block is still a trace body, and the
        tracer is what carries ``iter``/``all``, without which the block could
        only ever see a request's first forward — its prefill — and never the
        steps it generates::

            with model.register() as (tracer, registration):
                readouts = nnsight.save([])
                for step in tracer.all():
                    readouts.append(model.model.layers[16].output[0][-1])
        """
        # Mirror Tracer.__enter__ directly (not via super()) so capture and the
        # skip guard see the user's frame at the same depth, as EditingTracer does.
        self.capture()
        if skippable(self.node):
            skip_context(self)
        # Not id(self): this tracer is a temporary, so the next register() can be
        # allocated at the same address and collide with this one — which the
        # worker would read as a re-registration, overwriting the first block and
        # handing both handles the same values.
        self.registration = Registration(
            self._model, id=f"registration-{uuid.uuid4().hex}"
        )
        return self, self.registration

    def execute(self, code: CodeType) -> None:
        """Ship the block to the workers rather than running it here.

        ``copy`` is left off: the worker builds a fresh mediator per request from
        this template, so each request already has a scope of its own, and the
        block's saves have to land in it for collection to find them.
        """
        model = self._model
        if not model.dispatched:
            model.dispatch()
        _warn_if_prefix_caching(model)

        template = Mediator(
            code,
            self.info.frame.f_globals,
            dict(self.info.frame.f_locals),
            node=self.node,
        )
        registration = self.registration
        registration._rpc("nnsight_register", registration.id, dumps(template))
