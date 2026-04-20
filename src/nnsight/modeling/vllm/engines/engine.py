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
                # Worker returns ``{base_id: {var_name: value}}`` so the
                # step attaches each request's OWN saves sub-dict to its
                # OWN RequestOutput.  The previous ``ro.saves = saves``
                # (same dict bound to every finished output) entangled
                # concurrent separate traces whose user code used
                # overlapping variable names — one winner clobbered the
                # rest.  See Bug A in PP_DESIGN notes.
                saves_by_req = pickle.loads(saves_bytes)
                for ro in request_outputs:
                    if ro.finished:
                        per_req = saves_by_req.get(ro.request_id)
                        if per_req:
                            ro.saves = per_req

        return request_outputs
