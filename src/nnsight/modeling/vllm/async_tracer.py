from typing import Callable

from ...modeling.mixins.remoteable import RemoteInterleavingTracer


class AsyncInterleavingTracer(RemoteInterleavingTracer):
    """Tracer for async vLLM generation.

    Overrides ``execute()`` to serialize mediators into sampling params
    and store the prepared ``(prompts, params, kwargs)`` on the tracer
    instance rather than triggering synchronous generation via
    ``model.interleave()``.  The :class:`AsyncVLLMBackend` picks up the
    prepared data from the tracer after execution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared = None

    def execute(self, fn: Callable):
        # Run the compiled user code to set up mediators.
        fn(self.info, self)

        args = self.batcher.batched_args
        kwargs = self.batcher.batched_kwargs

        self.batcher.batched_args = tuple()
        self.batcher.batched_kwargs = {}

        interleaver = self.model._interleaver
        interleaver.initialize(self.mediators, self, batcher=self.batcher)

        # Dispatch the model if needed (loads weights).
        if not self.model.dispatched:
            self.model.dispatch()

        # Serialize mediators into the sampling params.
        prompts, params = self.model._prepare_generation(*args, **kwargs)
        self.prepared = (prompts, params, kwargs)

        self.mediators.clear()
