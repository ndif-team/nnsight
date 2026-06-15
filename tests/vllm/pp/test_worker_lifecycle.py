"""PP worker-lifecycle unit tests (no GPU).

The readiness gate releases a forward for request-iteration ``k`` based on the
worker's lifecycle state, modelled as a phase (``LEADING`` / ``PAST_LOCAL`` /
``TERMINATED``) plus the monotonic ``worker_iteration``. ``AT_LOCAL`` (the worker
blocked in ``request`` on a local module) is observed by the gate as
``parked``. These exercise ``PPWorkerProgress.is_ahead_of`` directly — the
predicate whose under-specified corners produced the historical PP hangs.
"""

import pytest

from nnsight.intervention.interleaver import PPWorkerProgress, WorkerPhase


def _progress(*, phase=WorkerPhase.LEADING, worker_iteration=0, max_iteration=0):
    p = PPWorkerProgress()
    p.phase = phase
    p.worker_iteration = worker_iteration
    p.max_iteration = max_iteration
    return p


class TestGatePredicate:
    """``is_ahead_of(k, parked=...)`` — one row per spec condition."""

    def test_fresh_worker_leading_makes_forward_wait(self):
        # Worker on iteration 0, not yet at its local part, not parked: the
        # forward for iteration 0 must wait — a local hook may still register.
        p = _progress(phase=WorkerPhase.LEADING, worker_iteration=0, max_iteration=8)
        assert p.is_ahead_of(0, parked=False) is False

    def test_parked_at_local_releases_current_iteration(self):
        # Blocked in request() on a local module: the hook is registered.
        p = _progress(phase=WorkerPhase.LEADING, worker_iteration=0, max_iteration=8)
        assert p.is_ahead_of(0, parked=True) is True

    def test_past_local_releases_current_iteration(self):
        # Went downstream this iteration: no local hooks remain.
        p = _progress(phase=WorkerPhase.PAST_LOCAL, worker_iteration=0, max_iteration=8)
        assert p.is_ahead_of(0, parked=False) is True

    def test_worker_past_iteration_releases(self):
        # Worker already ran iteration k, so iteration-k hooks were registered.
        p = _progress(phase=WorkerPhase.LEADING, worker_iteration=2, max_iteration=8)
        assert p.is_ahead_of(1, parked=False) is True

    def test_worker_behind_waits_even_if_parked(self):
        # Parked at iteration 0's local hook does NOT release iteration 1's
        # forward — the per-iteration settled signal only counts for k == it.
        p = _progress(phase=WorkerPhase.LEADING, worker_iteration=0, max_iteration=8)
        assert p.is_ahead_of(1, parked=True) is False
        p2 = _progress(phase=WorkerPhase.PAST_LOCAL, worker_iteration=0, max_iteration=8)
        assert p2.is_ahead_of(1, parked=False) is False

    def test_forward_past_worker_range_releases(self):
        # Single-shot trace (max_iteration 0): the forward for iteration 1 is
        # past anything this worker intervenes in.
        p = _progress(phase=WorkerPhase.LEADING, worker_iteration=0, max_iteration=0)
        assert p.is_ahead_of(1, parked=False) is True

    def test_terminated_releases_every_forward(self):
        # End / stop / error all land here: the worker will never register
        # another hook, for any k — including k inside [worker_iteration,
        # max_iteration], which no other condition covers (the errored-mid-
        # generation case).
        p = _progress(phase=WorkerPhase.TERMINATED, worker_iteration=2, max_iteration=8)
        assert p.is_ahead_of(2, parked=False) is True
        assert p.is_ahead_of(3, parked=False) is True
        assert p.is_ahead_of(7, parked=False) is True


class TestIterationReset:
    """``reset_iteration`` re-arms the per-iteration phase."""

    def test_reset_clears_settled_to_leading(self):
        # A settled phase from the previous iteration must not leak into the
        # next: after reset the forward for the new iteration waits again.
        p = _progress(phase=WorkerPhase.PAST_LOCAL, worker_iteration=0, max_iteration=8)
        p.reset_iteration(1)
        assert p.phase is WorkerPhase.LEADING
        assert p.worker_iteration == 1
        assert p.is_ahead_of(1, parked=False) is False

    def test_reset_clears_gone_remote(self):
        p = _progress(phase=WorkerPhase.PAST_LOCAL, worker_iteration=0)
        p.gone_remote = True
        p.reset_iteration(1)
        assert p.gone_remote is False
