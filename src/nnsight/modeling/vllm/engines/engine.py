from vllm.v1.engine.llm_engine import LLMEngine

from ..collect import merge_collected_saves


class NNsightLLMEngine(LLMEngine):
    """Custom vLLM engine that collects saved intervention results from finished requests.

    After each engine step, finished requests are forwarded to the
    model runner's ``finish_nnsight()`` method to gather any variables
    that were ``.save()``-ed during intervention execution.
    """

    def step(self):

        request_outputs = super().step()

        finished_req_ids = [ro.request_id for ro in request_outputs if ro.finished]

        if finished_req_ids:
            results = self.engine_core.collective_rpc(
                "collect_nnsight",
                args=(finished_req_ids, finished_req_ids),
            )
            # Assemble each rank's partial saves into one result (PP stages ship
            # sentinels for slots they don't own; merged position-wise — see
            # ``collect.merge_collected_saves``).
            merged = merge_collected_saves(results)
            if merged:
                for ro in request_outputs:
                    if ro.finished:
                        per_req = merged.get(ro.request_id)
                        if per_req:
                            ro.saves = per_req

        return request_outputs
