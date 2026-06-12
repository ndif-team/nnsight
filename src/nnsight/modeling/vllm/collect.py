"""Merge the per-rank save payloads returned by ``collect_nnsight``.

A single home for what used to be triplicated (identically in the async backend
and the sync engine, and *incorrectly* — a flat ``dict.update`` that clobbers PP
sentinels — in the serve handler).

Each ``collective_rpc("collect_nnsight", ...)`` result is one entry per rank:
``None`` for ranks that ship nothing (TP siblings), else a zstd-compressed
pickle of ``{base_id: {var_name: value}}``. With PP > 1 every PP stage's
TP-rank-0 contributes, each shipping only the slots it owns (foreign slots are
:data:`NOT_ON_THIS_RANK` sentinels), so same-named saves must be merged
position-wise with :func:`merge_saved` — never overwritten. For a single rank
(or scalar leaves) this reduces to the prior "later value wins".
"""

import pickle

import zstandard as _zstd

from .lazy_remote_tensor import merge_saved

_ZSTD_DECOMPRESSOR = _zstd.ZstdDecompressor()


def merge_collected_saves(results) -> dict:
    """Decompress, unpickle, and PP-merge per-rank ``collect_nnsight`` results.

    Returns ``{base_id: {var_name: value}}`` assembled across ranks.
    """
    merged: dict = {}
    for r in results:
        if r is None:
            continue
        rank_saves = pickle.loads(_ZSTD_DECOMPRESSOR.decompress(r))
        for base_id, per_req in rank_saves.items():
            dst = merged.setdefault(base_id, {})
            for name, value in per_req.items():
                if name not in dst:
                    dst[name] = value
                elif name == "__nnsight_exceptions__":
                    # The deferred-error envelope ships from EVERY rank whose
                    # worker raised (the same user error raises on each PP
                    # stage), with per-rank tracebacks that legitimately
                    # differ (device names, addresses). Same later-rank-wins
                    # union as merge_saved, minus the divergence tripwire.
                    dst[name] = {**dst[name], **value}
                else:
                    dst[name] = merge_saved(dst[name], value, label=name)
    return merged
