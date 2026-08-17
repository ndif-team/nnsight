"""Carry saved values back out of the engine.

A worker's saved values live in the worker process, and nothing in vLLM's own
output carries them. The engine knows when a request finishes, which is when its
worker has nothing left to run, so that is where they are fetched and attached to
the output the request produced.

Every finished request is collected for, not only the traced ones: a registered
block (see [`nnsight.modeling.vllm.registration`][nnsight.modeling.vllm.registration])
runs on requests nnsight never created, and this is the one place their values
can be handed back on the output they belong to.
"""

from __future__ import annotations

import pickle
from typing import Any

from vllm.v1.engine.llm_engine import LLMEngine


def merge_collected(payloads: list) -> dict:
    """Combine what each rank returned from ``collect_nnsight``.

    Not "take the first non-empty": a trace's values come from the rank holding
    the sampled output, while a registered block's come from whichever rank ran
    the layers it read — under pipeline parallelism those are different ranks,
    and taking one would silently drop the other.
    """
    merged: dict[str, dict] = {}
    for payload in payloads or ():
        if payload is None:
            continue
        for request_id, entry in pickle.loads(payload).items():
            into = merged.setdefault(
                request_id, {"saves": {}, "error": None, "registered": {}}
            )
            into["saves"].update(entry.get("saves") or {})
            into["registered"].update(entry.get("registered") or {})
            if into["error"] is None:
                into["error"] = entry.get("error")
    return merged


def attach(output: Any, entry: dict) -> None:
    """Put a request's collected values on its output.

    ``saves`` carries both kinds, since from the caller's side they are simply
    what was saved for this request. ``nnsight_saves`` keeps the trace's own
    apart, because that is what gets pushed back into the trace's variables and a
    registered value must not land there — see
    [`collect_nnsight`][nnsight.modeling.vllm.model_runners.GPUModelRunner.NNsightGPUModelRunner.collect_nnsight].
    """
    output.saves = {**entry["registered"], **entry["saves"]}
    output.nnsight_saves = entry["saves"]
    output.nnsight_error = entry["error"]


class NNsightLLMEngine(LLMEngine):
    """An engine that attaches each finished request's saved values to its output."""

    def step(self) -> Any:
        outputs = super().step()

        finished = [output.request_id for output in outputs if output.finished]
        if not finished:
            return outputs

        # The outputs ride along so a block parked on `tracer.result` can be
        # served: the value it is waiting for is assembled here, not in the worker.
        by_id = {output.request_id: output for output in outputs if output.finished}
        collected = merge_collected(
            self.engine_core.collective_rpc(
                "collect_nnsight", args=(finished, finished, by_id)
            )
        )
        for output in outputs:
            entry = collected.get(output.request_id)
            if entry is not None:
                attach(output, entry)

        return outputs
