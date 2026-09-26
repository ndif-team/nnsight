"""The scheduler of an engine whose stages return their values through it.

Under pipeline parallelism a stage before the last needs, at a request's next
step, the values the later stages served in this one. When the engine runs
with async scheduling, the last stage broadcasts them to the other stages
right after sampling, beside vLLM's own broadcast of the sampled tokens (see
the runner). When it does not (the Ray executor, which returns the sampled
tokens through the scheduler instead), the stages before the last never run
that broadcast, so the values take the tokens' route: the last stage puts
them on its output, this scheduler keeps them by request, and the request's
next scheduler output carries them to every stage. A request that finishes
has no next output, and what it left is dropped with it.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

BACKWARD = "nnsight_backward"


class NNsightScheduler(Scheduler):
    """vLLM's scheduler, carrying each request's pending cross-stage values."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Request id -> entries the last stage served for it in its latest
        # step, until the request's next step carries them out.
        self._nnsight_backward: dict[str, list[tuple]] = {}
        self._nnsight_pp = self.parallel_config.pipeline_parallel_size > 1

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> Any:
        # Only the last stage sets this, and only for a step that served something.
        entries = getattr(model_runner_output, BACKWARD, None)
        if entries:
            for entry in entries:
                self._nnsight_backward.setdefault(entry[2], []).append(entry)
        return super().update_from_output(scheduler_output, model_runner_output)

    def schedule(self) -> Any:
        output = super().schedule()
        if self._nnsight_pp:
            output.nnsight_backward = [
                entry for req in output.num_scheduled_tokens for entry in self._nnsight_backward.pop(req, [])
            ]
            for req in [req for req in self._nnsight_backward if req not in self.requests]:
                del self._nnsight_backward[req]
        return output
