"""Stream a vLLM trace's outputs as the async engine generates them.

An async engine (``VLLM(..., mode="async")``) runs its own output-handler loop
rather than a synchronous ``step()``, so a trace on it is consumed as an async
stream rather than returning once at the end::

    with model.trace("The Eiffel Tower is in", max_tokens=5) as tracer:
        logits = model.logits.save()

    async for output in tracer.backend:
        ...                        # a RequestOutput per decode step
    last.saves["logits"]           # saved values arrive on the *finished* output

Every yielded output carries a ``saves`` dict; only the finished one is non-empty.
``await tracer.backend`` drains the stream and returns just the last (finished)
output, when you don't need the intermediate steps::

    last = await tracer.backend
    last.saves["logits"]

This backend is what the tracer runs on ``__exit__`` and what ``tracer.backend``
iterates. On ``__exit__`` it builds and serializes the trace's workers and submits
the request to the engine, keeping the returned async generator; iterating it
streams each step's output and, on the finished one, fetches the request's saved
values from the worker and attaches them (re-raising a real intervention error).

Two caveats. The stream is **single-consumption** — the underlying engine generator
is consumed once, so iterate (or ``await``) it exactly once. And an **abort** (a
consumer that stops early) frees the request's worker in ``__aiter__``'s ``finally``,
which runs on ``aclose()`` — explicit, or when the generator is garbage-collected;
a bare ``break`` therefore defers the free to GC. To free promptly, ``aclose()`` it.

Interventions run in the worker exactly as in the synchronous path — the same
[`NNsightGPUModelRunner`][nnsight.modeling.vllm.model_runners.GPUModelRunner.NNsightGPUModelRunner],
the same per-request scoping. Only the collection of saved values differs: without
a ``step()`` to hook, it happens here, in the stream.
"""

from __future__ import annotations

import pickle
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator

from ...tracing.backend import Backend

if TYPE_CHECKING:
    from vllm import RequestOutput

    from .vllm import VLLM


class AsyncVLLMBackend(Backend):
    """Backend for a trace on an async [`VLLM`][nnsight.modeling.vllm.vllm.VLLM]."""

    def __init__(self, model: "VLLM") -> None:
        self.model = model
        self._generator: Any = None
        self._request_id: str | None = None

    def __call__(self, tracer: Any) -> None:
        """Build and submit the request; keep the engine's output stream.

        Runs on the trace's ``__exit__`` while the caller's frame is still live, so
        all frame-dependent work (building the workers, reading the block) happens
        here; the returned async generator is iterated later, once the frame is gone.
        """
        if not self.model.dispatched:
            self.model.dispatch()

        interleaver = self.model.interleaver
        try:
            # Build the workers and combined input without running a forward — the
            # engine runs it — then serialize the workers onto the request.
            _, call_args, forward_kwargs = tracer.prepare(tracer.info.code)
            prompts, params, lora_requests = call_args
            if len(prompts) != 1:
                raise NotImplementedError(
                    "Async tracing takes a single prompt; use one tracer.invoke or a "
                    "direct input, not several invokes."
                )
            self.model._attach_mediators(params, **forward_kwargs)

            self._request_id = str(uuid.uuid4())
            self._generator = self.model.vllm_entrypoint.generate(
                prompts[0],
                params[0],
                self._request_id,
                lora_request=lora_requests[0],
            )
        finally:
            # The serialized workers ride on the request now; the live ones are done.
            interleaver.cancel()

    async def __aiter__(self) -> "AsyncGenerator[RequestOutput, None]":
        """Yield each step's output; attach saved values to the finished one."""
        finished = False
        try:
            async for output in self._generator:
                # Every output carries a `saves` dict for a uniform shape; only the
                # finished one is ever non-empty (saves arrive once the block is done).
                output.saves = {}
                if output.finished:
                    finished = True
                    await self._attach_saves(output)
                yield output
        finally:
            # A stream closed before its finished output (the consumer stopped
            # iterating) aborts the request, which then never reaches the collection
            # that frees its worker — so free it here instead. Runner-side cleanup on
            # the engine's finished-id list cannot do this: it cannot tell an aborted
            # request from one that has finished but whose saves the client has not
            # collected yet, and would drop the latter's saves.
            if not finished:
                await self._free_worker()

    async def _free_worker(self) -> None:
        """Release an aborted request's worker (and its saved tensors) on the engine."""
        if self._request_id is None:
            return
        await self.model.vllm_entrypoint.collective_rpc(
            "collect_nnsight", args=([self._request_id], [self._request_id])
        )

    def __await__(self):
        """``await tracer.backend`` drains the stream and returns the last output."""

        async def _drain():
            last = None
            async for output in self:
                last = output
            return last

        return _drain().__await__()

    async def _attach_saves(self, output: "RequestOutput") -> None:
        """Fetch this finished request's saved values from the worker onto ``output``."""
        from ...intervention.errors import raise_deferred

        from .engines.engine import attach, merge_collected

        request_id = self._request_id
        results = await self.model.vllm_entrypoint.collective_rpc(
            "collect_nnsight",
            args=([request_id], [request_id], {request_id: output}),
        )
        # Merged rather than taking the first rank to answer: a trace's values
        # come from the rank holding the sampled output, a registered block's from
        # whichever rank ran the layers it read.
        entry = merge_collected(results).get(request_id)
        if entry is not None:
            attach(output, entry)
            raise_deferred(entry["error"])
