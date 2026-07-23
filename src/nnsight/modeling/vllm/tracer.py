"""A tracer whose workers can be built without running the model.

The base :class:`~nnsight.intervention.tracer.InterleavingTracer` builds a trace's
workers and, in the same step, runs the forward. An async vLLM trace instead hands
the request to the engine and streams the outputs, so it needs the worker-building
step on its own, before any forward. :class:`VLLMTracer` factors that step into
:meth:`prepare`; the synchronous path (:meth:`execute`) still runs it and then the
forward, exactly as the base does. Keeping this in the vLLM package leaves the base
tracer untouched.
"""

from __future__ import annotations

from types import CodeType
from typing import Any

from ...intervention.batching import Batcher
from ...intervention.interleaver import Mediator
from ...intervention.tracer import InterleavingTracer
from ...tracing.tracer import push_result
from ...tracing.util import Scope


class VLLMTracer(InterleavingTracer):
    """An :class:`InterleavingTracer` whose worker-building is callable on its own."""

    def prepare(self, code: CodeType) -> tuple:
        """Build the trace's workers and combined call input, without running the model.

        The first half of :meth:`execute`: collect the invoke workers (or the single
        direct-input worker) onto the interleaver and assemble the batched call
        input, then return the workers alongside it. The async backend uses this to
        get the workers to serialize and the input to submit, in place of
        :meth:`~nnsight.intervention.envoy.Envoy.interleave`.

        Returns:
            ``(workers, args, kwargs)`` — the workers to read results from, and the
            combined ``(args, kwargs)`` for the model call.
        """
        frame = self.info.frame
        glbls = frame.f_globals
        interleaver = self.envoy.interleaver
        # The batcher belongs to this trace; each Invoker adds its input to it through
        # self.tracer.batcher while the body runs to collect invokes.
        self.batcher = Batcher(self.envoy)
        # >0 rows means direct input (one implicit invoke); 0 means invoke mode
        # (the body defines the batch via tracer.invoke(...)). Trace-level params
        # that aren't data go to the call.
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
            forward_kwargs: dict[str, Any] = {}
        else:
            forward_kwargs = dict(self.kwargs)
            # Invokers append their workers as this runs.
            exec(code, Scope(dict(frame.f_locals), frame.f_locals, glbls))
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
            self.envoy.interleave(self.fn, *args, **kwargs)
            for mediator in mediators:
                push_result(self.info.frame, mediator.lcls)
        finally:
            # interleave clears the interleaver on its way out; do it here too so a
            # failure before it doesn't leave workers/batcher behind.
            self.envoy.interleaver.cancel()
