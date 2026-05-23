import pickle

import zstandard as _zstd

from vllm.v1.engine.llm_engine import LLMEngine

_ZSTD_DECOMPRESSOR = _zstd.ZstdDecompressor()


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
            # Worker returns ``{base_id: {var_name: value}}``. Without PP
            # only TP-rank-0 returns data and the merge is a single dict.
            # With PP > 1 every PP stage's TP-rank-0 contributes — merge
            # per-request sub-dicts so saves from different stages
            # accumulate. Later-rank-wins on duplicate names matches the
            # design's "owning rank wins" rule for cross-stage values.
            merged: dict = {}
            for r in results:
                if r is None:
                    continue
                rank_saves = pickle.loads(_ZSTD_DECOMPRESSOR.decompress(r))
                for base_id, per_req in rank_saves.items():
                    merged.setdefault(base_id, {}).update(per_req)
            if merged:
                for ro in request_outputs:
                    if ro.finished:
                        per_req = merged.get(ro.request_id)
                        if per_req:
                            ro.saves = per_req

        return request_outputs
