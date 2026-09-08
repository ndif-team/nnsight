"""Pipeline-parallel behaviors on one real PP=2 engine.

Every test runs on the session's shared engine (``pp2_engine``): a trace
written against the client's meta tree, executed across two stages. The tests
cover the read, write, and save paths, then each shape of ``tracer.iter``
against the cross-stage pull machinery. Where the failure mode of a shape is a
pull that never completes (a 30s timeout), the test bounds the elapsed time
well inside that limit.
"""

import time

import nnsight
import pytest
import torch

from _support import EARLY, LATE, PROMPT, PROMPT_B, STALL_BOUND_S

pytestmark = pytest.mark.gpu


def _layer(model, index):
    return model.model.layers[index]


# ---------------------------------------------------------------------------
# Reads, writes, saves
# ---------------------------------------------------------------------------


def test_cross_stage_reads_and_logits(pp2_engine):
    model = pp2_engine
    with model.trace(PROMPT, temperature=0.0, max_tokens=1):
        early = _layer(model, EARLY).output.save()
        late = _layer(model, LATE).output.save()
        logits = model.logits.save()

    # Layer outputs are (hidden, residual) tuples; every slot is a real tensor
    # after the merge (a sentinel would mean a stage's contribution was dropped).
    for name, value in (("early", early), ("late", late)):
        hidden = value[0] if isinstance(value, tuple) else value
        assert isinstance(hidden, torch.Tensor), (name, type(value))
        assert torch.isfinite(hidden.float()).all(), name
    assert isinstance(logits, torch.Tensor) and logits.shape[-1] > 100_000 // 2
    assert model.tokenizer.decode(logits[-1].argmax(dim=-1)).strip() == "Paris"


def test_cross_stage_write_changes_logits(pp2_engine):
    model = pp2_engine
    with model.trace(PROMPT, temperature=0.0, max_tokens=1):
        clean = model.logits.save()
    with model.trace(PROMPT, temperature=0.0, max_tokens=1):
        # A stage-0-owned write, replicated on both ranks: applied by the owner,
        # absorbed by the non-owner.
        hidden = _layer(model, EARLY).output[0]
        _layer(model, EARLY).output = (hidden * 0,) + tuple(_layer(model, EARLY).output[1:])
        zeroed = model.logits.save()
    assert not torch.equal(clean, zeroed)


def test_bounded_loop_saves_each_step(pp2_engine):
    model = pp2_engine
    with model.trace(PROMPT, temperature=0.0, max_tokens=4) as tracer:
        steps = nnsight.save([])
        for _ in tracer.iter[:4]:
            steps.append(_layer(model, LATE).output[0])
    assert len(steps) == 4
    assert all(isinstance(step, torch.Tensor) for step in steps)


def test_cache_unions_both_stages(pp2_engine):
    model = pp2_engine
    with model.trace(PROMPT, temperature=0.0, max_tokens=1) as tracer:
        cache = tracer.cache(modules=[_layer(model, EARLY), _layer(model, LATE)]).save()
    assert sorted(cache.keys()) == [f"model.model.layers.{EARLY}", f"model.model.layers.{LATE}"]


# ---------------------------------------------------------------------------
# tracer.iter shapes against the pull machinery
# ---------------------------------------------------------------------------


def test_open_loop_after_a_pre_loop_save_runs_every_step(pp2_engine):
    """A save before the loop parks the worker while the first step's gate
    serve passes; the loop then rides the remaining steps' serves."""
    model = pp2_engine
    with model.trace(PROMPT, temperature=0.0, max_tokens=6) as tracer:
        control = nnsight.save([])
        for step in tracer.iter[:]:
            control.append(step)
    with model.trace(PROMPT, temperature=0.0, max_tokens=6) as tracer:
        logits = model.logits.save()
        steps = nnsight.save([])
        for step in tracer.iter[:]:
            steps.append(step)
    assert len(steps) >= 6, list(steps)
    assert list(steps) == list(control)[: len(steps)]


def test_two_cross_stage_reads_per_step_match_across_traces(pp2_engine):
    """The second read of a pinned step's body names the same round as the
    first, whatever the engine's history: the same block returns the same
    values on a fresh and on a warmed engine."""
    model = pp2_engine

    def run():
        with model.trace(PROMPT, temperature=0.0, max_tokens=3) as tracer:
            vals = nnsight.save([])
            for _ in tracer.iter[:3]:
                a = _layer(model, LATE).output[0]
                b = _layer(model, LATE + 1).output[0]
                vals.append(float(a.sum()) + float(b.sum()))
        return list(vals)

    first, second = run(), run()
    assert len(first) == 3 and first == second, (first, second)


def test_bounded_loop_over_an_upstream_layer_completes(pp2_engine):
    """On the downstream rank the body's force is an upstream pull; a force for
    a round this rank has not opened parks and is served once that round runs."""
    model = pp2_engine
    t0 = time.time()
    with model.trace(PROMPT, temperature=0.0, max_tokens=4) as tracer:
        hs = nnsight.save([])
        for _ in tracer.iter[:4]:
            hs.append(float(_layer(model, EARLY).output[0].sum()))
    assert len(hs) == 4 and time.time() - t0 < STALL_BOUND_S


def test_open_loop_with_per_step_pulls_ends_with_the_run(pp2_engine):
    model = pp2_engine
    t0 = time.time()
    with model.trace(PROMPT, temperature=0.0, max_tokens=4) as tracer:
        hs = nnsight.save([])
        for _ in tracer.iter[:]:
            hs.append(float(_layer(model, LATE).output[0].sum()))
    assert len(hs) == 4 and time.time() - t0 < STALL_BOUND_S


def test_bounded_loop_past_generation_end_keeps_reached_steps(pp2_engine):
    """A loop asking for more steps than the request generates keeps the
    reached steps' values; the pulls it parked for rounds that never ran are
    unwound at collect, not waited on. The engine worker warns about the cut
    loop in its own process, so the client sees the values and no error."""
    model = pp2_engine
    t0 = time.time()
    with model.trace(PROMPT, temperature=0.0, max_tokens=4) as tracer:
        hs = nnsight.save([])
        for _ in tracer.iter[:8]:
            hs.append(float(_layer(model, LATE).output[0].sum()))
    assert len(hs) == 4 and time.time() - t0 < STALL_BOUND_S


def test_per_step_pulls_in_both_directions(pp2_engine):
    """Each step forces one value from each stage, so at generation end both
    ranks hold a parked pull of the other stage."""
    model = pp2_engine
    t0 = time.time()
    with model.trace(PROMPT, temperature=0.0, max_tokens=4) as tracer:
        pairs = nnsight.save([])
        for _ in tracer.iter[:]:
            early = float(_layer(model, EARLY).output[0].sum())
            late = float(_layer(model, LATE).output[0].sum())
            pairs.append((early, late))
    assert len(pairs) == 4 and time.time() - t0 < STALL_BOUND_S


def test_a_finished_request_leaves_a_concurrent_one_running(pp2_engine):
    """Collect for the short invoke's request serves only that request's
    workers; the long invoke's per-step pulls continue to their own end."""
    model = pp2_engine
    t0 = time.time()
    with model.trace(temperature=0.0) as tracer:
        with tracer.invoke(PROMPT, max_tokens=2):
            short_logits = model.logits.save()
        with tracer.invoke(PROMPT_B, max_tokens=8):
            vals = nnsight.save([])
            for _ in tracer.iter[:8]:
                x = _layer(model, LATE).output[0]
                y = _layer(model, LATE + 1).output[0]
                vals.append(float(x.sum()) + float(y.sum()))
    assert len(vals) == 8 and time.time() - t0 < STALL_BOUND_S
    assert isinstance(short_logits, torch.Tensor)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_every_rank_releases_finished_workers(pp2_engine):
    """After the traces above, no rank tracks a request: workers, their saved
    tensors, and their pull records go with the requests that finished."""
    counts = pp2_engine.vllm_entrypoint.llm_engine.collective_rpc("nnsight_request_count")
    assert counts and all(count == 0 for count in counts), counts
