"""Regression: the PP cross-rank pull must size its transfer from the
per-request *scheduled* token count, not from ``mediator.batch_group``,
because ``batch_group`` is legitimately repurposed mid-step.

Background (Window A). On every PP rank, each ``execute_model`` step:

  1. ``process_batch_groups`` sets ``mediator.batch_group = [start, N]`` —
     N = the request's scheduled token count (the leading dim of every
     layer activation in this forward pass).
  2. the forward pass runs; a free-running mediator on a non-owning rank
     may reach a remote-layer access and build a ``LazyRemoteTensor`` whose
     pull size is taken from ``mediator.batch_group[1]``.
  3. ``unflatten`` rewrites ``mediator.batch_group = [start, 1]`` — the
     *logits* are one row per request, so this is correct for sampling but
     wrong as a token count.

If the mediator reads ``batch_group`` after step 3, the pull sizes its recv
buffer for 1 token while the producer ships the real N-token activation ->
gloo "Received data size doesn't match" -> dead worker (single-token) or a
hung condition wait (multi-token).

These tests drive the *real* ``process_batch_groups`` / ``unflatten`` with
lightweight fakes (no GPU, no vLLM engine) and assert that an authoritative
per-request token count survives ``unflatten``. They use deliberately
non-gpt2 request ids and uneven prompt lengths so nothing is hard-coded to a
single batch shape.
"""

import pytest

from nnsight.modeling.vllm.model_runners.GPUModelRunner import (
    NNsightGPUModelRunner,
)

Helper = NNsightGPUModelRunner.NNsightRequestHelper


class _Med:
    """Stand-in for a Mediator: only the attributes the helper touches.

    Mirrors the real ``Mediator``'s unconditionally-initialized PP gate field
    ``_pp_scheduled_count`` (0 in ``Mediator.__init__``); the helper increments
    it per scheduled step via direct attribute access.
    """

    def __init__(self):
        self.batch_group = None
        self._pp_scheduled_count = 0


class _Batcher:
    def __init__(self):
        self.last_batch_group = "unset"


class _Interleaver:
    def __init__(self, mediators):
        self.mediators = mediators
        self.batcher = _Batcher()


class _Model:
    def __init__(self, mediators):
        self.interleaver = _Interleaver(mediators)


def _setup(scheduled):
    """Wire a helper + fake model for an ordered ``{req_id: num_tokens}``."""
    helper = Helper()
    mediators = {rid: _Med() for rid in scheduled}
    helper.mediators = dict(mediators)
    model = _Model(list(mediators.values()))
    helper._batch_req_ids = list(scheduled)
    helper._num_scheduled_tokens = dict(scheduled)
    return helper, model, mediators


def _pull_token_count(mediator):
    """The value the PP pull uses to size the cross-rank transfer.

    Mirrors what ``pp_envoy._pp_lazy_access`` reads when it builds the
    ``LazyRemoteTensor`` pull closure. Must equal the producer's buffered
    leading dim (the scheduled token count), regardless of ``unflatten``.
    """
    return getattr(mediator, "pp_num_tokens", None)


def test_pull_count_survives_unflatten_for_prefill():
    # Two concurrent prefills of *different* lengths — exactly the batched
    # case that crashed (11- and 5-token prompts).
    scheduled = {"decoder_req_alpha": 11, "decoder_req_beta": 5}
    helper, model, meds = _setup(scheduled)

    helper.process_batch_groups(scheduled, list(scheduled), model)
    # Forward-pass view: token-level slices are correct.
    assert meds["decoder_req_alpha"].batch_group == [0, 11]
    assert meds["decoder_req_beta"].batch_group == [11, 5]

    helper.unflatten(model)
    # unflatten legitimately rewrites batch_group to prompt-level [start, 1]
    # (one logits row per request). This is Window A: the field a late pull
    # would read is now 1, not the real token count.
    assert meds["decoder_req_alpha"].batch_group == [0, 1]
    assert meds["decoder_req_beta"].batch_group == [1, 1]

    # The authoritative per-request token count must still be recoverable so
    # the pull sizes its recv buffer to match the producer's N-token buffer.
    assert _pull_token_count(meds["decoder_req_alpha"]) == 11
    assert _pull_token_count(meds["decoder_req_beta"]) == 5


def test_pull_count_correct_for_decode_step():
    # Decode: a single new token. batch_group[1] == 1 is *correct* here, but
    # the authoritative count must independently report 1 too.
    scheduled = {"blocks_req_gamma": 1}
    helper, model, meds = _setup(scheduled)
    helper.process_batch_groups(scheduled, list(scheduled), model)
    helper.unflatten(model)
    assert _pull_token_count(meds["blocks_req_gamma"]) == 1


def test_unscheduled_mediator_count_is_cleared():
    # A registered mediator whose request isn't scheduled this step must not
    # leak a stale token count into a later pull.
    helper = Helper()
    med = _Med()
    med.batch_group = [3, 9]          # stale from an earlier step
    helper.mediators = {"layers_req_delta": med}
    model = _Model([med])
    helper._batch_req_ids = []
    helper._num_scheduled_tokens = {}

    helper.process_batch_groups({}, [], model)

    assert med.batch_group is None
    assert _pull_token_count(med) is None
