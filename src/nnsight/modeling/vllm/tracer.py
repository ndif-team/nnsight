"""A tracer whose workers can be built without running the model.

The base [`InterleavingTracer`][nnsight.intervention.tracer.InterleavingTracer] builds a trace's
workers and, in the same step, runs the forward. An async vLLM trace instead hands
the request to the engine and streams the outputs, so it needs the worker-building
step on its own, before any forward. [`VLLMTracer`][nnsight.modeling.vllm.tracer.VLLMTracer] factors that step into
[`prepare`][nnsight.modeling.vllm.tracer.VLLMTracer.prepare]; the synchronous path (`execute`) still runs it and then the
forward, exactly as the base does. Keeping this in the vLLM package leaves the base
tracer untouched.

It also carries what a group's block reads about its step,
[`group`][nnsight.modeling.vllm.tracer.VLLMTracer.group] (see
[`groups`][nnsight.modeling.vllm.groups]).
"""

from __future__ import annotations

from types import CodeType
from typing import Any

from ...intervention.interleaver import Mediator
from ...intervention.tracer import InterleavingTracer
from ...tracing.tracer import push_result
from ...intervention.util import clear_shared_locals, shared_locals
from ...tracing.util import Scope
from .groups import GROUP, Member


def no_barrier(n: int) -> None:
    """Refuse a barrier, which this runtime cannot hold.

    A barrier releases when ``n`` of a trace's blocks have reached it, which
    needs them all running against one forward. Here each invoke is a separate
    vLLM request, scheduled independently and possibly in different steps
    entirely, so the blocks never coexist and the barrier would simply never
    release — a hang rather than an error. Say so at the call instead.
    """
    raise NotImplementedError(
        f"tracer.barrier({n}) cannot work on vLLM: each invoke is its own "
        "request and the engine schedules them independently, so the blocks "
        "never run against the same forward and the barrier would never "
        "release. Compute the shared value outside the trace, or put both "
        "prompts in one invoke."
    )


class VLLMTracer(InterleavingTracer):
    """An [`InterleavingTracer`][nnsight.intervention.tracer.InterleavingTracer] whose worker-building is callable on its own."""

    def barrier(self, n: int) -> None:
        """Not available here — see [`no_barrier`][nnsight.modeling.vllm.tracer.no_barrier]."""
        no_barrier(n)

    @property
    def group(self) -> tuple[Member, ...]:
        """The layout of the step a group's block is in: who is in it, and where their rows are.

        A group — ``tracer.invoke([p1, ..., pn])`` — is served, at each step, the
        rows of the members the engine scheduled into that step, concatenated in
        member order. This says which members those are and how the value splits:
        one [`Member`][nnsight.modeling.vllm.groups.Member] per member in the step,
        in member order, giving its index in the list, the start and size of its
        rows in a module value, and its own step (0 its prefill, k its k-th decode
        step). Row ``i`` of ``model.logits`` and ``model.samples`` is entry
        ``i``'s. A member that has finished is not in it.

        Read it anywhere in the step: before the step has begun — at the top of a
        ``tracer.iter`` iteration — it waits for the step, like a read of the
        step's values would.

        Returns:
            A tuple of ``Member(index, start, size, step)``.

        Raises:
            ValueError: In a block that is not a group's (one prompt per invoke).

        Examples:
            The last token of every member, at every step::

                >>> with model.trace(temperature=0.0, max_tokens=4) as tracer:  # doctest: +SKIP
                ...     with tracer.invoke(["The Eiffel Tower is in", "Rome is in"]):
                ...         layouts = list().save()
                ...         for step in tracer.iter[:]:
                ...             members = tracer.group
                ...             h = model.transformer.h[8].output
                ...             h[[m.start + m.size - 1 for m in members]] += v
                ...             layouts.append(members)
        """
        mediator = Mediator.current("tracer.group")
        if not hasattr(mediator, "layout"):
            raise ValueError(
                "tracer.group is a group's layout, and this block has one prompt; a "
                "group is a list of prompts in one tracer.invoke([...])"
            )
        # Pinned to a step whose start the model has not reached: the block is
        # ahead of it, at the top of a `tracer.iter` iteration, and that step's
        # members are not chosen yet. Anywhere else the step is under way.
        iteration = mediator.iteration
        if iteration is not None and mediator.occurrence(GROUP) <= iteration:
            Mediator.value(GROUP)
        return mediator.layout

    def prepare(self, code: CodeType) -> tuple:
        """Build the trace's workers and combined call input, without running the model.

        The first half of `execute`: collect the invoke workers (or the single
        direct-input worker) onto the interleaver and assemble the batched call
        input, then return the workers alongside it. The async backend uses this to
        get the workers to serialize and the input to submit, in place of
        [`interleave`][nnsight.intervention.envoy.Envoy.interleave].

        Returns:
            ``(workers, args, kwargs)`` — the workers to read results from, and the
            combined ``(args, kwargs)`` for the model call.
        """
        frame = self.info.frame
        glbls = frame.f_globals
        interleaver = self.envoy.interleaver
        # The batcher belongs to this trace; each Invoker adds its input to it through
        # self.tracer.batcher while the body runs to collect invokes.
        self.batcher = self.envoy._batcher_class(self.envoy, self.kwargs)
        # >0 rows means direct input (one implicit invoke); 0 means invoke mode
        # (the body defines the batch via tracer.invoke(...)). Trace-level params
        # that aren't data go to the call.
        if self.envoy._batch_size(*self.args, **self.kwargs):
            mediator = Mediator(
                code,
                glbls,
                dict(frame.f_locals),
                node=self.node,
            )
            mediator.batch_group = self.batcher.add(*self.args, **self.kwargs)
            interleaver.mediators.append(mediator)
            forward_kwargs: dict[str, Any] = {}
        else:
            forward_kwargs = dict(self.kwargs)
            # Invokers append their workers as this runs.
            exec(code, Scope(dict(frame.f_locals), shared_locals(frame), glbls))
            if not interleaver.mediators:
                raise ValueError(
                    "trace() needs an input, or at least one "
                    "`with tracer.invoke(...)` block"
                )

        mediators = list(interleaver.mediators)
        args, kwargs = self.batcher.assemble(self.fn)
        return mediators, args, {**kwargs, **forward_kwargs}

    def execute(self, code: CodeType) -> None:
        """Build the workers, run the forward interleaved, push results back."""
        try:
            mediators, args, kwargs = self.prepare(code)
            try:
                self.envoy.interleave(self.fn, *args, **kwargs)
            finally:
                # Pushed even when the run raises. A block's error is deferred to
                # the collect and re-raised out of `interleave`, so pushing only
                # on success would hand back a frame with none of the values that
                # did arrive — including the ones from the invokes that finished.
                for mediator in mediators:
                    push_result(self.info.frame, mediator.lcls.writes())
        finally:
            # interleave clears the interleaver on its way out; do it here too so a
            # failure before it doesn't leave workers/batcher behind.
            self.envoy.interleaver.cancel()
            # The invoke bodies that shared a store are done with it; holding it
            # until the outermost trace exits leaks a frame per trace.
            clear_shared_locals()
