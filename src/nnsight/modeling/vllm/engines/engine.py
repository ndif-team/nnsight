import pickle
from vllm.v1.engine.llm_engine import LLMEngine


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
            # results is a list (one per worker). Rank-0 returns pickled bytes, others None.
            saves_bytes = next((r for r in results if r is not None), None)
            if saves_bytes:
                saves = pickle.loads(saves_bytes)
                for ro in request_outputs:
                    if ro.finished:
                        ro.saves = saves

        return request_outputs
