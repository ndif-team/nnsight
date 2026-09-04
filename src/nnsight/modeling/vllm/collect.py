"""Assemble the per-request save payloads of one trace into its results.

Locally, the invoke blocks of one trace share their outer names: a container
bound above the invokes (``rows = [None] * n; rows.save()``) is ONE object
every invoke mutates. On the engine each invoke rides its own request into the
worker process with its own COPY, so the copies come back one per request,
each carrying only that request's writes. :func:`merge_shared_saves`
reproduces the shared-object semantics before results are pushed back into the
trace frame.
"""

from __future__ import annotations


def merge_shared_saves(mediators: list, per_request_saves: list) -> dict:
    """Merge same-name saves across a multi-invoke trace's requests, in place.

    A name that was bound and saved ABOVE the invokes (``mediator.presaved``) is
    one object locally, so its per-request copies merge element-wise — same-length lists slot-wise with ``None`` as the
    unwritten marker (the common pattern: a pre-sized slot list each invoke
    fills at its own index), dicts by key-union, everything else (and any
    conflicting slot several requests wrote) keeping the later request's copy —
    and the merged value is written into every mediator's scope so results push
    identically whichever mediator they're read from.

    A name each block saved for itself is NOT merged, however many requests
    shipped it: those are separate values, not copies of one, and merging them
    would keep only the last, leaving three prompts holding one prompt's
    activation. The caller returns those as a list instead.

    Returns the merged shared values by name (the caller re-marks them saved:
    the merge builds new containers the save-gate hasn't seen).
    """

    def merge_pair(a, b):
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return [
                merge_pair(x, y) if not (x is None or y is None)
                else (y if x is None else x)
                for x, y in zip(a, b)
            ]
        if isinstance(a, dict) and isinstance(b, dict):
            return {**a, **b}
        return b

    presaved: set = set()
    for mediator in mediators:
        presaved |= getattr(mediator, "presaved", set())

    counts: dict = {}
    merged: dict = {}
    for saves in per_request_saves:
        for name, value in saves.items():
            if name not in presaved:
                continue
            counts[name] = counts.get(name, 0) + 1
            merged[name] = merge_pair(merged[name], value) if name in merged else value
    shared = {name: value for name, value in merged.items() if counts[name] > 1}
    for mediator in mediators:
        for name, value in shared.items():
            mediator.lcls[name] = value
    return shared
