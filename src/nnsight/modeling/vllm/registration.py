"""Interventions that stay on the engine and run for every request.

A trace ships its block to the worker on the request it rides, so a sweep of a
thousand prompts serializes and deserializes the same block a thousand times, and
only requests that *are* nnsight traces are touched at all.

A registration inverts that. The block goes to the worker once and stays there;
from then on every request the engine runs gets its own copy of it — including
requests submitted by something that has never heard of nnsight, an OpenAI-API
client on the same server. Each copy keeps its own scope, so what it saves is
that request's own, and it comes back on that request's ``RequestOutput`` —
``output.saves`` — by the same collect a traced value uses, then is dropped.

Example:
    >>> model = VLLM("meta-llama/Llama-3.1-8B", dispatch=True,
    ...              enable_prefix_caching=False)
    >>>
    >>> with model.register() as (tracer, registration):        # doctest: +SKIP
    ...     hidden = model.model.layers[16].output[0].save()
    >>>
    >>> outputs = model.vllm_entrypoint.generate(prompts, sampling)  # doctest: +SKIP
    >>> outputs[5].saves["hidden"].shape                        # doctest: +SKIP
    >>>
    >>> registration.clear()                                    # doctest: +SKIP

The block is written exactly like a trace body — the same envoy tree, the same
``.save()``. What it cannot do is anything that belongs to one particular
request: there is no prompt to invoke, so ``tracer.invoke(...)`` has no meaning
here, and the block applies to whatever the engine happens to run.

The engine has to be built with ``enable_prefix_caching=False``. A prefix-cached
token is served without a forward pass, so no hook fires for it and the block
sees fewer rows than the prompt has, with nothing to say so. A trace asks for its
own request to be recomputed; a registration rides requests it did not create and
cannot, so the cache has to be off at the engine — registering against one that
has it on warns.
"""

from __future__ import annotations

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
    live until `clear`.

    There is nothing to read here. What a registered block saves comes back on
    the ``RequestOutput`` of the request it ran on, as ``output.saves``, by the
    same collect that carries a traced value — so the values arrive where the
    request that produced them already is, and are dropped as they go. Nothing
    accumulates for as long as somebody is reading the outputs, which on the
    synchronous engine is every request there is.

    Synchronous engines only: installing the block is a ``collective_rpc``, which
    on ``mode="async"`` is a coroutine there is no safe way to finish from a
    plain statement, so `VLLM.register` refuses there rather than silently not
    installing it.

    Attributes:
        model: The engine this is registered on.
        id: The engine-wide name for this registration, used to address it in the
            worker.
    """

    def __init__(self, model: "VLLM", id: str) -> None:
        self.model = model
        self.id = id
        self.cleared = False

    def _rpc(self, method: str, *args: Any) -> list:
        """Reach every worker.

        Synchronous, because registering and clearing are ordinary statements.
        The async engine's ``collective_rpc`` is a coroutine instead, and there is
        no safe way to finish one from here — driving it on a loop of our own
        deadlocks against the loop the engine is already running on. So this
        refuses rather than returning a coroutine nobody awaits, which is what it
        used to do: the block was never installed and nothing said so.
        """
        import inspect

        engine = self.model.vllm_entrypoint
        # The sync entrypoint keeps the engine one level down; the async one is
        # already the engine.
        core = getattr(engine, "llm_engine", engine)
        result = core.collective_rpc(method, args=args)
        if inspect.iscoroutine(result):
            result.close()
            raise NotImplementedError(
                f"{method} has to reach the engine's workers, and on an async "
                "engine (mode='async') that call is a coroutine this cannot "
                "await. Registration is synchronous-engine only for now; build "
                "with mode='sync', or trace the requests you want to instrument."
            )
        return result

    async def _arpc(self, method: str, *args: Any) -> list:
        """`_rpc` from inside a running event loop, where awaiting is possible.

        Which is the only place it is: the engine's machinery binds to whichever
        loop first drives it, so a loop of our own on another thread never gets an
        answer (``run_coroutine_threadsafe`` times out), and the engine exposes no
        loop to target instead.
        """
        import inspect

        engine = self.model.vllm_entrypoint
        core = getattr(engine, "llm_engine", engine)
        result = core.collective_rpc(method, args=args)
        if inspect.isawaitable(result):
            return await result
        return result

    def clear(self) -> None:
        """Stop running the block and drop anything it has not handed back.

        Use `aclear` on an async engine, whose workers can only be reached from
        inside the loop.
        """
        if self.cleared:
            return
        self._rpc("nnsight_clear_registered", self.id)
        self.cleared = True

    async def aclear(self) -> None:
        """`clear`, awaited — the async engine's form."""
        if self.cleared:
            return
        await self._arpc("nnsight_clear_registered", self.id)
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
    [`Registration`][nnsight.modeling.vllm.registration.Registration] to switch it
    off again with.
    """

    def __init__(self, model: "VLLM", *, backend: Backend | None = None) -> None:
        super().__init__(model, "__call__", backend=backend)
        self._model = model
        self.registration: Registration | None = None
        # Set by `execute`, sent by whichever exit ran.
        self._payload: bytes | None = None

    def __enter__(self) -> tuple["RegisteringTracer", Registration]:
        """Enter the block, binding the tracer and the handle that ends it.

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

    async def __aenter__(self) -> tuple["RegisteringTracer", Registration]:
        """`__enter__`, for ``async with`` — the form an async engine needs.

        Spelled out rather than delegating to `__enter__`: `capture` reads the
        caller's frame at a fixed depth, so going through another call would hand
        it this method's frame and it would find no ``with`` block at all.
        """
        self.capture()
        if skippable(self.node):
            skip_context(self)
        self.registration = Registration(
            self._model, id=f"registration-{uuid.uuid4().hex}"
        )
        return self, self.registration

    def __exit__(self, *exception: Any) -> bool:
        handled = super().__exit__(*exception)
        if self._payload is not None:
            self.registration._rpc(
                "nnsight_register", self.registration.id, self._payload
            )
            self._payload = None
        return handled

    async def __aexit__(self, *exception: Any) -> bool:
        """Capture as usual, then await the send.

        The block is prepared by `execute` either way; only the trip to the
        workers differs, and on an async engine that trip has to be awaited from
        inside the loop.
        """
        handled = super().__exit__(*exception)
        if self._payload is not None:
            await self.registration._arpc(
                "nnsight_register", self.registration.id, self._payload
            )
            self._payload = None
        return handled

    def execute(self, code: CodeType) -> None:
        """Prepare the block for the workers rather than running it here.

        Only prepared: the send happens in ``__exit__``/``__aexit__``, because
        whether it can be awaited depends on which of the two ran.

        It goes to every rank, so each builds the same per-request copies and
        their reads stay in step — which is what a sharded model's gathers need.

        ``copy`` is left off: the worker builds a fresh mediator per request from
        this template, so each request already has a scope of its own, and the
        block's saves have to land in it for the collect to find them.
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
        self._payload = dumps(template)
