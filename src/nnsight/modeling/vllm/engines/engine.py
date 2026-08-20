"""Carry saved values back out of the engine.

A worker's saved values live in the worker process, and nothing in vLLM's own
output carries them. The engine knows when a request finishes, which is when its
worker has nothing left to run, so that is where they are fetched and attached to
the output the request produced.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.engine.llm_engine import LLMEngine


class NNsightLLMEngine(LLMEngine):
    """An engine that attaches each finished request's saved values to its output."""

    def step(self) -> Any:
        outputs = super().step()

        finished = [output.request_id for output in outputs if output.finished]
        if not finished:
            return outputs

        results = self.engine_core.collective_rpc(
            "collect_nnsight", args=(finished, finished)
        )
        # Under PP every stage ships its own slots; merge them (a single-rank
        # run reduces to that rank's payload).
        from ..collect import merge_collected

        collected = merge_collected(results)
        if collected is None:
            return outputs

        for output in outputs:
            entry = collected.get(output.request_id)
            if entry is not None:
                output.saves = entry["saves"]
                output.nnsight_error = entry["error"]

        return outputs
