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
            # Merge saves from all PP ranks. Each rank returns pickled
            # saves for its local modules. Later ranks override on
            # duplicate keys (owning rank wins).
            all_saves = {}
            for r in results:
                if r is not None:
                    all_saves.update(pickle.loads(r))
            if all_saves:
                for ro in request_outputs:
                    if ro.finished:
                        ro.saves = all_saves

        return request_outputs
