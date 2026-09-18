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
from typing import Any, Optional

from vllm.v1.engine.llm_engine import LLMEngine

from ..pp_saved import fill_saves


def merge_collected(payloads: list) -> dict:
    """Combine what each rank returned from ``collect_nnsight``.

    Every rank runs the block and reports what it saved. The earliest rank's
    value of a name wins: under tensor parallelism the ranks hold equal values
    on different devices, and under pipeline parallelism the first stage's
    copy of a read it consumed is the one it computed with. A name the first
    stage saved as a placeholder (a read it only saved, of a location another
    stage holds; see `pp_saved`) is filled from the stage that holds it.
    Registered values are taken from every rank the same way, since a
    registered block runs wherever the layers it reads live.
    """
    merged: dict[str, dict] = {}
    reports: dict[str, dict] = {}
    for payload in payloads or ():
        if payload is None:
            continue
        for request_id, entry in pickle.loads(payload).items():
            into = merged.setdefault(
                request_id,
                {"saves": {}, "error": None, "registered": {}, "sequences": {}},
            )
            report = reports.setdefault(request_id, {"saves": [], "registered": [], "sequences": {}})
            report["saves"].append(entry.get("saves") or {})
            report["registered"].append(entry.get("registered") or {})
            for index, sequence in (entry.get("sequences") or {}).items():
                into["sequences"].setdefault(index, {"saves": {}, "registered": {}})
                per_index = report["sequences"].setdefault(index, {"saves": [], "registered": []})
                per_index["saves"].append(sequence.get("saves") or {})
                per_index["registered"].append(sequence.get("registered") or {})
            if into["error"] is None:
                into["error"] = entry.get("error")
    for request_id, report in reports.items():
        into = merged[request_id]
        try:
            into["saves"] = fill_saves(report["saves"])
            into["registered"] = fill_saves(report["registered"])
            for index, per_index in report["sequences"].items():
                into["sequences"][index]["saves"] = fill_saves(per_index["saves"])
                into["sequences"][index]["registered"] = fill_saves(per_index["registered"])
        except RuntimeError as error:
            # A placeholder nobody filled: the stage holding the location did
            # not reach that save. Its own error says why; failing that, this.
            if into["error"] is None:
                into["error"] = {"type_name": "RuntimeError", "message": str(error), "traceback": "", "is_control_flow": False}
    return merged


def attach(output: Any, entry: dict) -> None:
    """Put a request's collected values on its output.

    ``saves`` carries both kinds, since from the caller's side they are simply
    what was saved for this request. ``nnsight_saves`` keeps the trace's own
    apart, because that is what gets pushed back into the trace's variables and a
    registered value must not land there — see
    [`collect_nnsight`][nnsight.modeling.vllm.model_runners.GPUModelRunner.NNsightGPUModelRunner.collect_nnsight].

    A request that asked for several sampled sequences ran the block once per
    sequence, so each has values of its own. They go on the completion they
    belong to — ``output.outputs[i].saves``, alongside that sequence's text and
    token ids — while ``output.saves`` stays the primary sequence's, which is all
    there is unless ``n`` was set.
    """
    output.saves = {**entry["registered"], **entry["saves"]}
    output.nnsight_saves = entry["saves"]
    output.nnsight_error = entry["error"]

    sequences = entry.get("sequences") or {}
    for completion in getattr(output, "outputs", ()) or ():
        sequence = sequences.get(getattr(completion, "index", 0))
        if sequence is not None:
            completion.saves = {**sequence["registered"], **sequence["saves"]}
    # The trace's own, per sequence and in order, for the push back into the
    # caller's variables.
    output.nnsight_sequences = [
        sequences[index]["saves"] for index in sorted(sequences)
    ]


async def acollect(engine: Any, request_id: str, output: Any = None) -> Optional[dict]:
    """One finished request's collected values, merged across the ranks.

    The awaitable form, for the engines whose ``collective_rpc`` is a coroutine.
    ``output`` is the finished ``RequestOutput``, which the worker cannot build
    and needs in order to serve ``tracer.result``; without one this is simply the
    call that winds the request's workers up and frees what they held.
    """
    ids = [request_id]
    args = (ids, ids) if output is None else (ids, ids, pickle.dumps({request_id: output}))
    return merge_collected(await engine.collective_rpc("collect_nnsight", args=args)).get(
        request_id
    )


class NNsightLLMEngine(LLMEngine):
    """An engine that attaches each finished request's saved values to its output."""

    def step(self) -> Any:
        outputs = super().step()

        # The outputs ride along so a block parked on `tracer.result` can be
        # served: the value it is waiting for is assembled here, not in the worker.
        by_id = {output.request_id: output for output in outputs if output.finished}
        if not by_id:
            return outputs

        # Pickled here: the utility call is msgpack-encoded on its way to the
        # engine core, which refuses a plain object like RequestOutput unless
        # VLLM_ALLOW_INSECURE_SERIALIZATION is set; bytes pass natively.
        finished = list(by_id)
        collected = merge_collected(
            self.engine_core.collective_rpc(
                "collect_nnsight", args=(finished, finished, pickle.dumps(by_id))
            )
        )
        for output in outputs:
            entry = collected.get(output.request_id)
            if entry is not None:
                attach(output, entry)

        return outputs
