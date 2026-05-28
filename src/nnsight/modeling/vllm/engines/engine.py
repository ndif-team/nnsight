import pickle

import zstandard as _zstd

from vllm.v1.engine.llm_engine import LLMEngine

from ..lazy_remote_tensor import merge_saved

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
            # With PP > 1 every PP stage's TP-rank-0 contributes — each ships
            # only the slots it owns (others are NOT_ON_THIS_RANK sentinels),
            # so same-named saves are merged position-wise to assemble one
            # complete result (``merge_saved``). For scalars this reduces to
            # the prior "later-rank-wins" behavior.
            merged: dict = {}
            for r in results:
                if r is None:
                    continue
                rank_saves = pickle.loads(_ZSTD_DECOMPRESSOR.decompress(r))
                for base_id, per_req in rank_saves.items():
                    dst = merged.setdefault(base_id, {})
                    for name, value in per_req.items():
                        dst[name] = (
                            merge_saved(dst[name], value) if name in dst else value
                        )
            if merged:
                for ro in request_outputs:
                    if ro.finished:
                        per_req = merged.get(ro.request_id)
                        if per_req:
                            ro.saves = per_req

        return request_outputs
