"""Line one block up with another, mid-run.

The blocks of a trace — one per ``with tracer.invoke(x):`` — run in an order the
model chooses, not the order they were written: each resumes when the model
reaches the location *it* asked for. That is what makes them a batch rather than
a sequence, and it is fine as long as they don't need anything from each other.

They need a meeting point as soon as one hands something to another. A block that
reads an activation and puts it somewhere the next block takes it from is only
correct if the read happens first, and neither block can see the other's progress
to know. A :class:`Barrier` is that meeting point: every block that holds one
calls it, each waits, and the last to arrive releases them all — so everything
written above a barrier has happened before anything written below one.

Here two invokes share one value: the first reads a prompt's embeddings, the
second generates from them, and the barrier pins the read before the write so the
receiver isn't a :class:`NameError` waiting to happen.

.. code-block:: python

    from nnsight.modeling.transformers import TransformersModel

    model = TransformersModel("openai-community/gpt2", dispatch=True)

    with model.generate(max_new_tokens=3, do_sample=False) as tracer:
        barrier = tracer.barrier(2)

        with tracer.invoke("Madison Square Garden is in the city of"):
            embeddings = model.transformer.wte.output
            barrier()
            tokens = tracer.result.save()

        with tracer.invoke("_ _ _ _ _ _ _ _ _"):
            barrier()
            model.transformer.wte.output = embeddings

    # The underscore prompt carries none of the meaning; the embeddings do, so
    # both continuations end "New York City".

Waiting costs nothing but a greenlet switch — a block parks exactly as it does
for a value it is waiting on the model to produce.
"""

from __future__ import annotations

from .interleaver import Mediator


class Barrier:
    """A meeting point for the blocks of one trace.

    Built by :meth:`~nnsight.intervention.tracer.InterleavingTracer.barrier` with
    the number of blocks that will call it. Calling it parks the block until that
    many have arrived; the last one through releases the rest and none of them
    waits again. A barrier fewer blocks reach than it was built for never
    releases, and the blocks left waiting say so once the run ends.

    Attributes:
        n: How many blocks have to arrive before any of them continues.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        # The blocks parked here now. Emptied on release, so one barrier can be
        # used again — each round waits for its own n arrivals.
        self._waiting: list[Mediator] = []

    def __call__(self) -> None:
        """Wait here until every block this barrier waits for has arrived."""
        self._waiting.append(Mediator.current("tracer.barrier()"))
        if len(self._waiting) < self.n:
            Mediator.barrier()
            return
        # Last one in: let the others go, then carry on without waiting. Each
        # resumes here, runs to wherever it parks next, and hands control back —
        # the same trip it would make from the model side.
        waiting, self._waiting = self._waiting[:-1], []
        for other in waiting:
            other.pending = other.switch()
