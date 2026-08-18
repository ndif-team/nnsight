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
one engine request per invoke, keeping each returned async generator; iterating
streams every request's outputs in arrival order and, on each finished one,
fetches that request's saved values from the worker and attaches them
(re-raising a real intervention error).

A multi-invoke trace therefore yields one finished output per invoke, each
carrying that invoke's saves. A value saved above the invoke blocks ships back
once per request, each copy holding only that request's writes; once the last
request finishes, the copies are merged exactly as on the sync path
([`merge_shared_saves`][nnsight.modeling.vllm.collect.merge_shared_saves]) and
the merged values ride on the last finished output.

Two caveats. The stream is **single-consumption** — the underlying engine
generators are consumed once, so iterate (or ``await``) it exactly once. And an
**abort** (a consumer that stops early) frees the outstanding requests' workers
in ``__aiter__``'s ``finally``, which runs on ``aclose()`` — explicit, or when
the generator is garbage-collected; a bare ``break`` therefore defers the free
to GC. To free promptly, ``aclose()`` it.

Interventions run in the worker exactly as in the synchronous path — the same
[`NNsightGPUModelRunner`][nnsight.modeling.vllm.model_runners.GPUModelRunner.NNsightGPUModelRunner],
the same per-request scoping. Only the collection of saved values differs: without
a ``step()`` to hook, it happens here, in the stream.
"""

from __future__ import annotations

import asyncio
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
        # One (request_id, output stream) per invoke, in invoke order.
        self._streams: list[tuple[str, Any]] = []
        # The trace's workers, aligned with `_streams` — kept for the
        # shared-save merge once every request has finished.
        self._mediators: list = []
        # Per-request saves by request id, filled as requests finish.
        self._saves_by_request: dict[str, dict] = {}
        # Request ids whose collection RPC ran (their engine worker is wound up).
        self._collected: set[str] = set()

    def __call__(self, tracer: Any) -> None:
        """Build and submit one request per invoke; keep the output streams.

        Runs on the trace's ``__exit__`` while the caller's frame is still live, so
        all frame-dependent work (building the workers, reading the block) happens
        here; the streams are iterated later, once the frame is gone. Submitting
        immediately lets the engine start batching the requests before the user
        awaits.
        """
        if not self.model.dispatched:
            self.model.dispatch()

        interleaver = self.model.interleaver
        try:
            # Build the workers and combined input without running a forward — the
            # engine runs it — then serialize the workers onto the requests.
            _, call_args, forward_kwargs = tracer.prepare(tracer.info.code)
            prompts, params, lora_requests = call_args
            self._mediators = self.model._attach_mediators(params, **forward_kwargs)

            self._streams = []
            for prompt, param, lora_request in zip(prompts, params, lora_requests):
                request_id = str(uuid.uuid4())
                stream = self.model.vllm_entrypoint.generate(
                    prompt,
                    param,
                    request_id,
                    lora_request=lora_request,
                )
                self._streams.append((request_id, stream))
        finally:
            # The serialized workers ride on the requests now; the live ones are done.
            interleaver.cancel()

    async def __aiter__(self) -> "AsyncGenerator[RequestOutput, None]":
        """Yield every request's outputs as they arrive; attach saves to finished ones.

        Each stream is drained by its own pump task feeding one queue, so a
        multi-invoke trace's outputs interleave in generation order rather than
        request by request.
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def pump(stream: Any) -> None:
            try:
                async for output in stream:
                    queue.put_nowait(("output", output))
            except BaseException as error:  # re-raised on the consumer below
                queue.put_nowait(("error", error))
            finally:
                queue.put_nowait(("done", None))

        tasks = [asyncio.ensure_future(pump(stream)) for _, stream in self._streams]
        pending = len(tasks)
        try:
            while pending:
                kind, payload = await queue.get()
                if kind == "done":
                    pending -= 1
                    continue
                if kind == "error":
                    raise payload
                output = payload
                # Every output carries a `saves` dict for a uniform shape; only a
                # finished one is ever non-empty (saves arrive once the block is done).
                output.saves = {}
                if output.finished:
                    await self._attach_saves(output)
                yield output
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # A stream closed before its finished output (the consumer stopped
            # iterating, or another request's error ended the loop) aborts the
            # request, which then never reaches the collection that frees its
            # worker — so free it here instead. Runner-side cleanup on the
            # engine's finished-id list cannot do this: it cannot tell an aborted
            # request from one that has finished but whose saves the client has
            # not collected yet, and would drop the latter's saves.
            await self._free_workers()

    async def _free_workers(self) -> None:
        """Release uncollected requests' workers (and their saved tensors) on the engine."""
        remaining = [
            request_id
            for request_id, _ in self._streams
            if request_id not in self._collected
        ]
        if not remaining:
            return
        await self.model.vllm_entrypoint.collective_rpc(
            "collect_nnsight", args=(remaining, remaining)
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
        """Fetch one finished request's saved values from the worker onto ``output``.

        When this is the trace's last request to finish, first merge the copies
        of any save shared across invokes and lay the merged values onto this
        output, then re-raise the request's deferred error if it has one.
        """
        from ...intervention.errors import raise_deferred

        request_id = output.request_id
        results = await self.model.vllm_entrypoint.collective_rpc(
            "collect_nnsight", args=([request_id], [request_id])
        )
        self._collected.add(request_id)
        # Only the rank holding the sampled output returns anything.
        payload = next((result for result in results if result is not None), None)
        entry = (
            pickle.loads(payload).get(request_id) if payload is not None else None
        )
        if entry is not None:
            output.saves = entry["saves"]
        self._saves_by_request[request_id] = output.saves

        if len(self._collected) == len(self._streams):
            self._merge_shared(output)

        if entry is not None:
            raise_deferred(entry["error"])

    def _merge_shared(self, last_output: "RequestOutput") -> None:
        """Merge invoke-shared saves once every request is in; attach to ``last_output``.

        A container bound and saved above the invoke blocks came back once per
        request, each copy carrying only that request's writes. Merge them in
        invoke order — the same
        [`merge_shared_saves`][nnsight.modeling.vllm.collect.merge_shared_saves]
        the sync path runs — and put the merged values on the last finished
        output, which is where a consumer that drained the stream looks.
        """
        from .collect import merge_shared_saves

        per_request_saves = [
            self._saves_by_request.get(request_id, {})
            for request_id, _ in self._streams
        ]
        shared = merge_shared_saves(self._mediators, per_request_saves)
        if shared:
            last_output.saves = {**last_output.saves, **shared}
