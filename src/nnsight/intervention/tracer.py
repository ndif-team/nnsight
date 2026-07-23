"""The tracer behind ``with model.trace(...):``.

:class:`InterleavingTracer` is the context manager a user gets from
:meth:`Envoy.trace`. It captures the body of the ``with`` block instead of
running it inline, then — on exit — runs that body as intervention code
interleaved with the model's forward pass (see
:mod:`nnsight.intervention.interleaver`).

Concretely, the tracer turns::

    with envoy.trace(x):
        l1 = envoy.l1.output.save()

into: run ``envoy(x)`` while a worker executes the block, parking on
``envoy.l1.output`` until the model's ``l1`` module produces its output, then
resuming with that value. Saved values survive once the ``with`` exits.

The capture/compile/execute plumbing lives in the base
:class:`~nnsight.tracing.tracer.Tracer`; this module only overrides
:meth:`~InterleavingTracer.execute` to drive the interleaver, and adds the
trace-body API (:meth:`~InterleavingTracer.stop`,
:attr:`~InterleavingTracer.result`).
"""

from __future__ import annotations

from types import CodeType, TracebackType
from typing import TYPE_CHECKING, Any, Callable

import torch
from greenlet import getcurrent
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ..tracing.backend import Backend
from ..tracing.tracer import Tracer, push_result, save
from ..tracing.util import Scope, clean_traceback
from .barrier import Barrier
from .batching import Batcher
from .cache import Cache, CacheView
from .eproperty import eproperty
from .interleaver import EarlyStopException, Mediator
from .iterator import Iterations

if TYPE_CHECKING:
    from .envoy import Envoy


class InterleavingTracer(Tracer):
    """Runs a ``with`` block as interventions interleaved with a model call.

    Constructed by :meth:`Envoy.trace`; the ``fn`` and forward-pass arguments are
    remembered here and only executed when the ``with`` block exits, at which
    point :meth:`execute` hands the captured body to :meth:`Envoy.interleave` to
    run alongside ``fn(*args, **kwargs)`` (usually the model's forward).

    Inside the block, code reads and edits activations through :class:`Envoy`
    properties (``envoy.l1.output``, ``envoy.l2.input``, ...) and can use the
    tracer's own members:

    * :attr:`result` — the value ``fn`` returned (e.g. the model's output).
    * :meth:`stop` — halt the model run early.

    Attributes:
        envoy: The :class:`Envoy` whose interleaver and hooks this trace runs on.
        fn: The callable to run interleaved, or its name on the module. A name
            is left unresolved until :meth:`Envoy.interleave` binds it against the
            live module (e.g. after a lazy dispatch swaps in real weights).
        args: Positional arguments forwarded to ``fn`` (typically the input).
        kwargs: Keyword arguments forwarded to ``fn``.
    """

    def __init__(
        self,
        envoy: Envoy,
        fn: Callable | str,
        *args: Any,
        backend: Backend | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(backend=backend)
        self.envoy = envoy
        # fn stays unresolved (a name or callable) until interleave runs, so a
        # string is bound against the module that is actually live then (e.g.
        # after a lazy dispatch swaps in real weights).
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        # The batcher for this trace's invokes, created in execute(); Invokers add
        # their inputs to it through self.tracer.batcher. interleave() registers it
        # on the interleaver for the run so handle() can narrow per invoke.
        self.batcher: Batcher | None = None

    def stop(self) -> None:
        """Halt the model run as soon as this point is reached.

        Everything captured before the stop is kept; the model does not run past
        it, so later locations become unreachable::

            with model.trace("The Eiffel Tower is in") as tracer:
                hidden = model.transformer.h[0].output.save()
                tracer.stop()
        """
        raise EarlyStopException()

    @eproperty
    def result(self, value: Any) -> Any:
        """The value the traced call returned — e.g. the model's output.

        For :meth:`~nnsight.intervention.envoy.Envoy.generate` this is the token
        ids, so read it inside the block and ``.save()`` it to keep it::

            with model.generate("Madison Square Garden is in", max_new_tokens=3) as tracer:
                ids = tracer.result.save()
        """
        return value

    @property
    def iter(self) -> Iterations:
        """Target specific occurrences of a location across a repeated run.

        When the traced call reaches a module more than once — e.g. each step of
        a generation loop — looping over ``tracer.iter`` binds reads and writes in
        the loop body to a chosen range of those occurrences:

        .. code-block:: python

            with model.generate("...", max_new_tokens=10) as tracer:
                for step in tracer.iter[:3]:
                    hidden = model.transformer.h[0].output.save()  # steps 0, 1, 2

        See :class:`Iterations` for how the range is selected.
        """
        return Iterations()

    def barrier(self, n: int) -> Barrier:
        """A meeting point for ``n`` of this trace's blocks.

        The blocks of a trace run in the order the model reaches what each asked
        for, so one that hands something to another needs a point both agree on.
        Every block that holds the barrier calls it; the last to arrive releases
        them all, so everything above a barrier has happened before anything
        below one.

        Examples:
            >>> with model.generate(max_new_tokens=3) as tracer:
            ...     barrier = tracer.barrier(2)
            ...     with tracer.invoke("Madison Square Garden is in"):
            ...         embeddings = model.transformer.wte.output
            ...         barrier()
            ...     with tracer.invoke("_ _ _ _ _"):
            ...         barrier()
            ...         model.transformer.wte.output = embeddings

        Args:
            n: How many blocks will call this barrier. Fewer than this and it
                never releases; the blocks left waiting report it when the run
                ends.

        Returns:
            A :class:`~nnsight.intervention.barrier.Barrier` to call from each of
                those blocks.
        """
        return Barrier(n)

    def all(self) -> Iterations:
        """Every occurrence — a shorthand for ``tracer.iter[:]``.

        .. code-block:: python

            with model.generate("...", max_new_tokens=10) as tracer:
                for step in tracer.all():
                    hidden = model.transformer.h[0].output.save()  # every step
        """
        return self.iter[:]

    def cache(
        self,
        modules: list[Any] | None = None,
        device: Any = torch.device("cpu"),
        dtype: Any = None,
        detach: bool = True,
        include_output: bool = True,
        include_inputs: bool = False,
    ) -> CacheView:
        """Record activations of many modules at once during the run.

        Returns a :class:`~nnsight.intervention.cache.CacheView` that fills in as
        the model runs; read captured values from it after the trace::

            with model.trace(prompt) as tracer:
                cache = tracer.cache()                 # every module's output
            cache["model.transformer.h.0"].output      # by path
            cache.transformer.h[0].output              # or by navigation

        Only modules the run reaches *after* this call are captured. Values are
        recorded post-intervention. In a generation loop a module is captured once
        per step (``len(cache[path])`` is the step count).

        Args:
            modules: Envoys or path strings to capture; ``None`` captures every module.
            device: Device to move captured tensors to (default CPU); ``None`` leaves them.
            dtype: Optional dtype to cast captured tensors to.
            detach: Detach captured tensors from the autograd graph.
            include_output: Capture each module's output.
            include_inputs: Capture each module's inputs.
        """
        cache = Cache(
            self.envoy,
            modules=modules,
            device=device,
            dtype=dtype,
            detach=detach,
            include_output=include_output,
            include_inputs=include_inputs,
        )
        # The trace body runs in the worker greenlet; register the cache on its
        # mediator so the interleaver feeds it every location from here on.
        Mediator.current("tracer.cache()").caches.append(cache)
        # Save so the view survives past the trace (like a .save()d value). The
        # view is a virtual root above the model (see CacheView).
        return save(CacheView(cache, None))

    def invoke(self, *args: Any, **kwargs: Any) -> "Invoker":
        """Add a batched input inside a ``with model.trace() as tracer:`` block.

        Each ``with tracer.invoke(x):`` block contributes its input as one group of
        rows in a single combined forward, and its body's interventions see only
        those rows. Give ``trace()`` no input and use one ``invoke`` per prompt::

            with model.trace() as tracer:
                with tracer.invoke("the cat"):
                    a = model.transformer.h[0].output[0].save()
                with tracer.invoke("a much longer prompt here"):
                    b = model.transformer.h[0].output[0].save()

        An empty ``tracer.invoke()`` sees the whole batch (no row scoping).
        """
        return Invoker(self, *args, **kwargs)

    def execute(self, code: CodeType) -> None:
        """Run the captured trace body interleaved with the (possibly batched) model call.

        Two shapes, distinguished by whether ``trace()`` itself got input:

        * **direct input** (``trace(x)``) — the whole block is one implicit invoke;
          the block runs as a single worker over ``x``.
        * **invoke mode** (``trace()``) — the block is run now to collect its
          ``tracer.invoke(...)`` sub-blocks (each registers its input and captured
          body); their inputs are combined into one batched forward and each
          sub-block runs as a worker scoped to its rows.

        Either way the workers are handed to :meth:`Envoy.interleave` to run
        alongside ``fn(*combined_input)``, then results are pushed back with
        save-gating (see :meth:`Tracer.execute`).
        """
        frame = self.info.frame
        glbls = frame.f_globals
        # The batcher belongs to this trace; each Invoker adds its input to it
        # through self.tracer.batcher while the body runs to collect invokes.
        interleaver = self.envoy.interleaver
        # The model picks its batcher class (diffusion doubles for guidance);
        # the trace's forward kwargs (num_images_per_prompt, ...) reach it there.
        self.batcher = self.envoy._batcher_class(self.envoy, self.kwargs)
        try:
            # Is there actual input data, or only params? _batch_size reports the
            # number of data rows: >0 means direct input (one implicit invoke); 0
            # means invoke mode (the body defines the batch via tracer.invoke(...)).
            # Trace-level params that aren't data (max_new_tokens) go to the call.
            if self.envoy._batch_size(*self.args, **self.kwargs):
                mediator = Mediator(
                    code,
                    glbls,
                    dict(frame.f_locals),
                    node=self.node,
                    shared=frame.f_locals,
                )
                mediator.batch_group = self.batcher.add(*self.args, **self.kwargs)
                interleaver.mediators.append(mediator)
                forward_kwargs = {}
            else:
                forward_kwargs = dict(self.kwargs)
                # Invokers append their workers as this runs.
                exec(code, Scope(dict(frame.f_locals), frame.f_locals, glbls))
                if not interleaver.mediators:
                    raise ValueError(
                        "trace() needs an input, or at least one "
                        "`with tracer.invoke(...)` block"
                    )

            # The workers to push results from, snapshotted before interleave
            # prepends the whole-batch edit workers and cancel() clears the list.
            mediators = list(interleaver.mediators)
            # interleave assembles the batcher into the combined call; trace-level
            # params (e.g. max_new_tokens) ride alongside and win over assembled kwargs.
            self.envoy.interleave(self.fn, batcher=self.batcher, **forward_kwargs)
            for mediator in mediators:
                push_result(frame, mediator.lcls)
        finally:
            # interleave clears the interleaver on its way out; do it here too so a
            # failure before interleave doesn't leave workers/batcher behind.
            interleaver.cancel()

    def traceback(self, exception: BaseException) -> TracebackType | None:
        """Return a clean traceback for an error raised inside the trace body.

        When a worker raises, :meth:`Mediator.switch` stashes the intervention-only
        traceback on the exception as ``__intervention_tb__`` before the model and
        hook frames pile on top during unwinding. Prefer that stashed trace, fall
        back to the live one, and filter nnsight's own frames out so the user sees
        just their own code.
        """
        # Prefer the traceback captured at the worker boundary (clean,
        # intervention-only) over the one accumulated while unwinding.
        tb = getattr(exception, "__intervention_tb__", exception.__traceback__)
        return clean_traceback(tb)


class ScanningTracer(InterleavingTracer):
    """Like :class:`InterleavingTracer`, but runs the forward under fake tensors.

    Constructed by :meth:`~nnsight.modeling.mixins.meta.Meta.scan`. The block
    still reads activations through :class:`Envoy` properties, but the model runs
    inside a :class:`~torch._subclasses.fake_tensor.FakeTensorMode`, so operations
    only propagate tensor *metadata* (shape/dtype/device) — no real compute, no
    real weights. This lets you inspect activation shapes on an undispatched
    (meta-weight) model without loading it: :meth:`Meta.interleave` sees the active
    fake mode and skips dispatch, leaving parameters on the meta device, and
    :meth:`Envoy.interleave` moves the inputs onto that same device so shapes
    propagate consistently.

    The values read inside a scan are fake tensors, valid only within the block.
    Read their metadata (``.shape``, ``.dtype``) there; a fake tensor saved out of
    the scan cannot be used once the fake mode has exited.
    """

    def execute(self, code: CodeType) -> None:
        """Run :meth:`InterleavingTracer.execute` under a fake-tensor mode.

        Deferring to ``super().execute`` means scan goes through the same
        ``_batch_size``/``_batch`` preprocessing as trace (a string prompt is
        tokenized, invokes are batched); wrapping it in a
        :class:`~torch._subclasses.fake_tensor.FakeTensorMode` makes the forward
        propagate only shapes/dtypes — no real compute, no dispatch.
        ``assume_static_by_default`` keeps shapes concrete rather than symbolic, and
        ``allow_non_fake_inputs`` lets the meta-device parameters take part without
        being faked first.
        """
        with FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(assume_static_by_default=True),
        ):
            super().execute(code)


class Invoker(Tracer):
    """One ``with tracer.invoke(x):`` block inside a ``with model.trace() as tracer:``.

    Captures its body and registers it — together with the batch group its input
    occupies — on the parent :class:`InterleavingTracer`, which combines all
    invokes into a single batched forward (see :meth:`InterleavingTracer.execute`).
    """

    def __init__(self, tracer: "InterleavingTracer", *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if tracer.envoy.interleaver.interleaving:
            raise ValueError("Cannot invoke while the model is already running.")
        self.tracer = tracer
        self.args = args
        self.kwargs = kwargs

    def execute(self, code: CodeType) -> None:
        """Register this invoke's input and body, deferring the run to the parent.

        Adds the input to the parent tracer's :attr:`~InterleavingTracer.batcher`
        (claiming a batch group) and appends the captured body as a worker on the
        interleaver. The parent's :meth:`InterleavingTracer.execute` runs them all
        together once every invoke has been collected.
        """
        frame = self.info.frame
        interleaver = self.tracer.envoy.interleaver
        mediator = Mediator(
            code,
            frame.f_globals,
            dict(frame.f_locals),
            node=self.node,
            shared=frame.f_locals,
        )
        mediator.batch_group = self.tracer.batcher.add(*self.args, **self.kwargs)
        interleaver.mediators.append(mediator)
