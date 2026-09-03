
import pytest
import torch
import nnsight
import torch.nn as nn
from greenlet import getcurrent

from nnsight.intervention.barrier import Barrier
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import (
    Event,
    Interleaver,
    Mediator,
    OutOfOrderError,
    Pending,
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 8)
        self.l2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.l2(self.l1(x))


@pytest.fixture
def model():
    return Net()


@pytest.fixture
def envoy(model):
    return Envoy(model)


@pytest.fixture
def x():
    return torch.randn(2, 8)


def make_mediator(source, **scope):
    """A Mediator running `source` (a block) with `scope` as its locals.

    Mediator holds a captured block (code + globals/locals); these tests build one
    from a small source string, with Mediator/Event/getcurrent available as globals.
    """
    from textwrap import dedent

    code = compile(dedent(source), "<test-mediator>", "exec")
    glbls = {"Mediator": Mediator, "Event": Event, "getcurrent": getcurrent}
    return Mediator(code, glbls, scope)


def requester(store, location="loc"):
    """A mediator that requests `location` and stashes the value it receives."""
    return make_mediator(
        "store['got'] = Mediator.value(location)", store=store, location=location
    )


class TestMediator:
    def test_not_alive_before_start(self):
        med = requester({})
        assert not med.alive
        assert med.pending is None

    def test_alive_and_parked_after_start(self):
        med = requester({})
        med.start()
        assert med.alive
        # A park carries the occurrence tag for the iteration the worker wants
        # (0 by default).
        assert med.pending == Pending(Event.VALUE, "loc", 0)

    def test_handle_value_match_resumes_and_finishes(self):
        store = {}
        med = requester(store)
        med.start()
        med.handle("loc", 99)
        assert store["got"] == 99
        assert not med.alive
        assert med.pending is None

    def test_handle_value_no_match_stays_parked(self):
        store = {}
        med = requester(store)
        med.start()
        med.handle("other", 1)
        assert med.alive
        assert med.pending == Pending(Event.VALUE, "loc", 0)
        assert "got" not in store

    def test_handle_is_noop_when_done(self):
        med = requester({})
        med.start()
        med.handle("loc", 1)
        # second call with no live worker must not raise
        med.handle("loc", 2)
        assert not med.alive

    def test_value_helper_parks_on_value_event(self):
        # Mediator.value(loc) should park on (VALUE, loc, occurrence 0) and
        # return the served value back into the worker.
        store = {}
        med = make_mediator("store['got'] = Mediator.value('here')", store=store)
        med.start()
        assert med.pending == Pending(Event.VALUE, "here", 0)
        med.handle("here", 42)
        assert store["got"] == 42

    def test_event_tags_location_with_iteration(self):
        # Mediator.event(kind, location) records the worker's current iteration
        # alongside the location before switching to the parent.
        store = {}
        med = make_mediator(
            "store['got'] = Mediator.event(Event.VALUE, 'loc')", store=store
        )
        med.start()
        assert med.pending == Pending(Event.VALUE, "loc", 0)

    def test_pending_prints_as_location_and_occurrence(self):
        # The location and occurrence are separate fields so the model side can
        # match on either without building a string, but an error message wants
        # them rejoined — printing is what OutOfOrderError interpolates.
        assert str(Pending(Event.VALUE, "model.layers.16.output", 2)) == (
            "model.layers.16.output.i2"
        )
        assert f"{Pending(Event.SWAP, 'loc', 0, 'edit')}" == "loc.i0"

    def test_iteration_binds_read_to_that_occurrence(self):
        # With iteration set to 1, a read waits past the model's first visit to a
        # location and binds to the second.
        store = {}
        med = make_mediator(
            "getcurrent().mediator().iteration = 1\n"
            "store['got'] = Mediator.value('loc')",
            store=store,
        )
        # Driven through an Interleaver rather than by calling `med.handle`
        # directly: counting the visits is the interleaver's job (a worker parked
        # elsewhere is never asked, so it could not count them itself), and this
        # test is about which visit the read binds to.
        il = Interleaver()
        il.mediators.append(med)
        with il:
            assert med.pending == Pending(Event.VALUE, "loc", 1)
            il.handle("loc", "first")  # occurrence 0 — ignored, worker stays parked
            assert med.alive and "got" not in store
            il.handle("loc", "second")  # occurrence 1 — served
        assert store["got"] == "second"
        assert not med.alive


class TestInterleaver:
    def test_interleaving_flag_toggles(self):
        il = Interleaver()
        assert not il.interleaving
        with il:
            assert il.interleaving
        assert not il.interleaving

    def test_starts_mediators_on_enter(self):
        il = Interleaver()
        med = requester({})
        il.mediators.append(med)
        assert not med.alive
        with il:
            assert med.alive

    def test_handle_fans_out_to_all_mediators(self):
        il = Interleaver()
        s1, s2 = {}, {}
        il.mediators.append(requester(s1))
        il.mediators.append(requester(s2))
        with il:
            assert il.parked == {"loc"}
            il.handle("loc", 7)
        assert s1["got"] == 7
        assert s2["got"] == 7

    def test_cancel_clears_mediators(self):
        il = Interleaver()
        il.mediators.append(make_mediator("pass"))  # never started -> not dangling
        il.cancel()
        assert il.mediators == []

    def test_parked_set_is_rebuilt_each_run(self):
        # A prior run's wait count must not leak into the next run.
        il = Interleaver()
        first = requester({}, "a")
        il.mediators.append(first)
        with il:
            assert il.parked == {"a"}
        il.cancel()

        second = requester({}, "b")
        il.mediators.append(second)
        with il:
            assert il.parked == {"b"}

    def test_cache_routes_do_not_stick_across_runs(self):
        class Cache:
            def subscriptions(self):
                return {"a.output": ("a", "output")}

        il = Interleaver()
        first = requester({}, "a")
        il.mediators.append(first)
        with il:
            first.caches.append(Cache())
            il.reindex()
            assert "a.output" in il.observers
        il.cancel()

        second = requester({}, "b")
        il.mediators.append(second)
        with il:
            assert not il.observers

    def test_only_ready_waiters_are_served(self):
        il = Interleaver()
        target, other = requester({}, "target"), requester({}, "other")
        il.mediators.extend([target, other])
        called = []
        original = other.handle

        def spy(*args):
            called.append(args)
            return original(*args)

        other.handle = spy
        with il:
            il.handle("target", 1)

        assert not called
        assert other.occurrence("target") == 1

    def test_waiting_route_moves_with_a_worker(self):
        store = {}
        mediator = make_mediator(
            "store['a'] = Mediator.value('a')\n"
            "store['b'] = Mediator.value('b')",
            store=store,
        )
        il = Interleaver()
        il.mediators.append(mediator)

        with il:
            assert il.parked == {"a"}
            il.handle("a", 1)
            assert il.parked == {"b"}
            il.handle("b", 2)

        assert store == {"a": 1, "b": 2}

    @pytest.mark.xfail(
        reason="A visit is counted once, after everyone parked on it is served: a worker "
        "re-parking on a location during the same visit (released from a barrier) is "
        "served that visit's value again rather than waiting for the next visit.",
        strict=True,
    )
    def test_barrier_repark_uses_the_next_provider_occurrence(self):
        # The first worker has already had its current occurrence counted when
        # the second worker releases it from the barrier. Its next read of loc
        # must therefore wait for the next hook, not be stranded on occurrence 0.
        store = {}
        barrier = Barrier(2)
        first = make_mediator(
            "store['first'] = Mediator.value('loc')\n"
            "getcurrent().mediator().iteration = None\n"
            "barrier()\n"
            "store['after'] = Mediator.value('loc')",
            store=store,
            barrier=barrier,
        )
        second = make_mediator(
            "store['second'] = Mediator.value('loc')\nbarrier()",
            store=store,
            barrier=barrier,
        )
        il = Interleaver()
        il.mediators.extend([first, second])

        with il:
            il.handle("loc", 1)
            assert first.pending == Pending(Event.VALUE, "loc", 1)
            il.handle("loc", 2)

        assert store == {"first": 1, "second": 1, "after": 2}

    def test_barrier_receiver_written_first_is_served_in_the_same_visit(self):
        # The receiver is the *earlier* worker: parked on the barrier when loc is
        # first reached, released by the reader during that same visit, and only
        # then asking for loc. It has not consumed this occurrence, so it is
        # served in it — not stranded waiting for a next visit that never comes.
        store = {}
        barrier = Barrier(2)
        receiver = make_mediator(
            "barrier()\nstore['receiver'] = Mediator.value('loc')",
            store=store,
            barrier=barrier,
        )
        reader = make_mediator(
            "store['reader'] = Mediator.value('loc')\nbarrier()",
            store=store,
            barrier=barrier,
        )
        il = Interleaver()
        il.mediators.extend([receiver, reader])

        with il:
            il.handle("loc", 1)

        assert store == {"reader": 1, "receiver": 1}
        assert not receiver.alive and not reader.alive

    def test_targeted_cache_observes_only_its_subscription(self):
        class Cache:
            def __init__(self):
                self.observed = []

            def subscriptions(self):
                return {"target.output": ("target", "output")}

            def observe_selected(self, selected, value):
                self.observed.append((selected, value))

        il = Interleaver()
        mediator = make_mediator("pass")
        il.mediators.append(mediator)
        cache = Cache()

        with il:
            mediator.caches.append(cache)
            il.reindex()
            il.handle("other.output", 1)
            il.handle("target.output", 2)

        assert cache.observed == [(('target', 'output'), 2)]

    def test_check_dangling_throws_into_parked_worker(self):
        il = Interleaver()
        med = requester({}, "never-reached")
        il.mediators.append(med)
        with il:  # starts the mediator; it parks waiting for "never-reached"
            assert med.alive
            with pytest.raises(OutOfOrderError, match="never-reached"):
                il.check_dangling_mediators()


class TestHooks:
    def test_forward_passes_through_when_not_interleaving(self, envoy, model, x):
        assert not envoy.interleaver.interleaving
        out = model(x)
        assert torch.allclose(out, model.l2(model.l1(x)))


class TestEnvoyAccess:
    def test_output_captures_layer_output(self, envoy, model, x):
        captured = {}
        with envoy.trace(x):
            captured["l1"] = envoy.l1.output
        assert torch.allclose(captured["l1"], model.l1(x))

    def test_alias_reads_same_activation(self, model, x):
        # A renamed module reads the same activation through its alias.
        envoy = Envoy(model, rename={"l1": "first"})
        with envoy.trace(x):
            via_alias = envoy.first.output.save()
            via_name = envoy.l1.output.save()
        assert torch.equal(via_alias, via_name)

    def test_adhoc_module_application(self, envoy, model, x):
        # Calling an envoy inside a trace runs its module ad hoc; applying l2 to
        # l1's output reproduces the normal l2(l1(x)) flow.
        with envoy.trace(x):
            manual = envoy.l2(envoy.l1.output).save()
            real = envoy.output.save()
        assert torch.allclose(manual, real)

    def test_inputs_captures_layer_input(self, envoy, model, x):
        captured = {}
        with envoy.trace(x):
            captured["l2_in"] = envoy.l2.inputs
        args, kwargs = captured["l2_in"]
        assert torch.allclose(args[0], model.l1(x))

    def test_root_output(self, envoy, model, x):
        captured = {}
        with envoy.trace(x):
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model(x))

    def test_trace_clears_mediators_after_run(self, envoy, x):
        with envoy.trace(x):
            captured = nnsight.save(envoy.l1.output)
        # The intervention ran (captured a real value)...
        assert isinstance(captured, torch.Tensor)
        # ...and cancel() cleared the run's mediators afterward.
        assert envoy.interleaver.mediators == []

    def test_out_of_order_raises(self, envoy, x):
        with pytest.raises(OutOfOrderError) as exc:
            with envoy.trace(x):
                later = envoy.l2.output   # l2 runs after l1...
                earlier = envoy.l1.output  # ...so requesting l1 now is out of order
        assert "model.l1.output" in str(exc.value)

    def test_out_of_order_traceback_points_at_waiting_line(self, envoy, x, monkeypatch):
        # Assert the default (cleaned) traceback regardless of ambient DEBUG (which
        # -v / --verbose turns on, keeping the full stack).
        monkeypatch.setattr(nnsight.CONFIG.APP, "DEBUG", False)
        try:
            with envoy.trace(x):
                later = envoy.l2.output
                earlier = envoy.l1.output
        except OutOfOrderError as error:
            import traceback as tb_module

            summary = tb_module.extract_tb(error.__traceback__)[-1]
            assert summary.filename.endswith("test_interleaving.py")
            assert "envoy.l1.output" in summary.line
        else:
            pytest.fail("expected OutOfOrderError")

    def test_in_order_does_not_raise(self, envoy, model, x):
        captured = {}
        with envoy.trace(x):
            captured["l1"] = envoy.l1.output
            captured["l2"] = envoy.l2.output
        assert torch.allclose(captured["l1"], model.l1(x))


class TestEnvoyEditing:
    def test_set_output_replaces_it(self, envoy, model, x):
        replacement = torch.zeros(2, 8)
        captured = {}
        with envoy.trace(x):
            envoy.l1.output = replacement
            captured["out"] = envoy.output
        # l2 runs on the replaced l1 output.
        assert torch.allclose(captured["out"], model.l2(replacement))

    def test_edit_output_from_its_value(self, envoy, model, x):
        captured = {}
        with envoy.trace(x):
            envoy.l1.output = envoy.l1.output + 1
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.l2(model.l1(x) + 1))

    def test_inplace_edit_output(self, envoy, model, x):
        captured = {}
        with envoy.trace(x):
            envoy.l1.output[:] = 0
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.l2(torch.zeros(2, 8)))

    def test_set_input_replaces_it(self, envoy, model, x):
        replacement = torch.zeros(2, 8)
        captured = {}
        with envoy.trace(x):
            envoy.l2.input = replacement
            captured["out"] = envoy.output
        # l2 runs on the replaced input instead of l1's output.
        assert torch.allclose(captured["out"], model.l2(replacement))

    def test_pure_set_does_not_require_read(self, envoy, model, x):
        # A write with no prior read still lands (registered before the run).
        captured = {}
        with envoy.trace(x):
            envoy.l1.output = torch.ones(2, 8)
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.l2(torch.ones(2, 8)))


class Loop(nn.Module):
    """Calls `block` `steps` times, feeding each output back in (a decode loop)."""

    def __init__(self, steps=5):
        super().__init__()
        self.block = nn.Linear(8, 8)
        self.steps = steps

    def forward(self, x):
        outs = []
        for _ in range(self.steps):
            x = self.block(x)
            outs.append(x)
        return torch.stack(outs)


class SourceOpLoop(nn.Module):
    """Loops over an internal *operation* (not a submodule) inside one forward.

    Mirrors a Mixture-of-Experts expert loop: the single call site ``self.step``
    fires `inner` times per forward, so its source location counts invocations.
    Each invocation differs by `+ i` so per-invocation capture is verifiable.
    """

    def __init__(self, dim=8, inner=5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))
        self.inner = inner

    def step(self, x, i):
        return x @ self.weight + i

    def forward(self, x):
        out = torch.zeros_like(x)
        for i in range(self.inner):
            out = out + self.step(x, i)
        return out


class TestIteration:
    @pytest.fixture
    def loop(self):
        torch.manual_seed(0)
        return Loop(steps=5)

    @pytest.fixture
    def loop_envoy(self, loop):
        return Envoy(loop)

    def test_iter_binds_reads_to_each_step(self, loop, loop_envoy, x):
        ref = loop(x)  # (steps, ...) — one block output per step
        with loop_envoy.trace(x) as tracer:
            captured = nnsight.save([])
            for step in tracer.iter[:3]:
                captured.append(loop_envoy.block.output)
        for i, got in enumerate(captured):
            assert torch.allclose(got, ref[i])

    def test_iter_offset_slice(self, loop, loop_envoy, x):
        ref = loop(x)
        with loop_envoy.trace(x) as tracer:
            captured = nnsight.save([])
            for step in tracer.iter[2:5]:
                captured.append(loop_envoy.block.output)
        for j, got in enumerate(captured):
            assert torch.allclose(got, ref[j + 2])

    def test_plain_read_binds_to_first_occurrence(self, loop, loop_envoy, x):
        ref = loop(x)
        with loop_envoy.trace(x):
            first = nnsight.save(loop_envoy.block.output)
        assert torch.allclose(first, ref[0])

    def test_iter_swap_targets_one_step(self, loop, loop_envoy, x):
        ref = loop(x)
        with loop_envoy.trace(x) as tracer:
            for step in tracer.iter[1:2]:
                loop_envoy.block.output = torch.zeros(2, 8)
            result = nnsight.save(tracer.result)
        assert torch.count_nonzero(result[1]).item() == 0
        assert torch.allclose(result[0], ref[0])  # step 0 untouched

    def test_iteration_restored_after_loop(self, loop, loop_envoy, x):
        # After a `tracer.iter` loop, iteration returns to its prior value (0), so
        # a following read binds to the first occurrence — but here step 0 has
        # already run, making it out of order.
        with pytest.raises(OutOfOrderError):
            with loop_envoy.trace(x) as tracer:
                for step in tracer.iter[:2]:
                    loop_envoy.block.output
                loop_envoy.block.output  # back to .i0, already ran past -> raises

    def test_unbounded_iter_runs_every_step_then_warns(self, loop, loop_envoy, x):
        # `iter[:]` keeps going until the model stops producing steps; the trailing
        # request for the step that never runs warns rather than raising, and the
        # values from every real step are kept.
        ref = loop(x)  # 5 steps
        with pytest.warns(UserWarning, match="never reached"):
            with loop_envoy.trace(x) as tracer:
                captured = nnsight.save([])
                for step in tracer.iter[:]:
                    captured.append(loop_envoy.block.output)
        assert len(captured) == 5
        for i, got in enumerate(captured):
            assert torch.allclose(got, ref[i])

    def test_all_is_shorthand_for_open_iter(self, loop, loop_envoy, x):
        # tracer.all() == tracer.iter[:] — runs every step, warns on overrun.
        ref = loop(x)
        with pytest.warns(UserWarning, match="never reached"):
            with loop_envoy.trace(x) as tracer:
                captured = nnsight.save([])
                for step in tracer.all():
                    captured.append(loop_envoy.block.output)
        assert len(captured) == 5
        for i, got in enumerate(captured):
            assert torch.allclose(got, ref[i])

    def test_iter_relaxes_after_first_hit(self, loop, loop_envoy, x):
        # Inside a pinned step, the first read binds to that step; a later read of
        # the same path is relaxed and binds to the same (current) occurrence.
        ref = loop(x)
        with loop_envoy.trace(x) as tracer:
            got = nnsight.save({})
            for step in tracer.iter[2:3]:  # single step, pinned to 2
                got["a"] = loop_envoy.block.output  # pinned -> occ 2
                got["b"] = loop_envoy.block.output  # relaxed -> occ 2
        assert torch.allclose(got["a"], ref[2])
        assert torch.allclose(got["b"], ref[2])

    def test_iter_read_then_swap_same_step(self, loop, loop_envoy, x):
        # A read then a write of the same location in one pinned step both land on
        # that step's occurrence (the write is relaxed but stays on the current
        # occurrence).
        ref = loop(x)
        with loop_envoy.trace(x) as tracer:
            for step in tracer.iter[1:2]:
                loop_envoy.block.output = loop_envoy.block.output + 1
            result = nnsight.save(tracer.result)
        assert torch.allclose(result[1], ref[1] + 1)
        assert torch.allclose(result[0], ref[0])  # step 0 untouched

    def test_iter_single_int(self, loop, loop_envoy, x):
        ref = loop(x)
        with loop_envoy.trace(x) as tracer:
            captured = nnsight.save([])
            for step in tracer.iter[2]:
                captured.append(loop_envoy.block.output)
        assert len(captured) == 1
        assert torch.allclose(captured[0], ref[2])

    def test_iter_sparse_list(self, loop, loop_envoy, x):
        # Skipped occurrences (1, 3) still advance the count, so 2 and 4 line up.
        ref = loop(x)
        with loop_envoy.trace(x) as tracer:
            got = nnsight.save([])
            for step in tracer.iter[[0, 2, 4]]:
                got.append((step, loop_envoy.block.output))
        assert [s for s, _ in got] == [0, 2, 4]
        for s, captured in got:
            assert torch.allclose(captured, ref[s])

    def test_negative_iter_raises(self, loop, loop_envoy, x):
        with pytest.raises(ValueError, match="negative"):
            with loop_envoy.trace(x) as tracer:
                for step in tracer.iter[-1:2]:
                    loop_envoy.block.output

    def test_iter_past_the_end_warns_and_keeps_saved_values(self, loop, loop_envoy, x):
        # A loop that asks for a step the run does not make is cut short there:
        # what it saved is kept, the statements after it never run, and the only
        # signal is the warning. Reading `tail` afterwards is the caller's error.
        tail = None
        with pytest.warns(UserWarning, match="was never reached"):
            with loop_envoy.trace(x) as tracer:  # the model runs 5 steps
                captured = nnsight.save([])
                for step in tracer.iter[:8]:
                    captured.append(loop_envoy.block.output)
                tail = nnsight.save("after the loop")
        assert len(captured) == 5
        assert tail is None  # the statement after the loop never ran

    def test_bounded_iter_matching_the_run_keeps_its_tail(self, loop, loop_envoy, x):
        # The other side of the same rule: a bound the run does reach leaves the
        # loop to end on its own, so the statements after it run.
        with loop_envoy.trace(x) as tracer:
            captured = nnsight.save([])
            for step in tracer.iter[:5]:
                captured.append(loop_envoy.block.output)
            tail = nnsight.save("after the loop")
        assert len(captured) == 5
        assert tail == "after the loop"

    def test_sparse_iter_past_the_end_warns(self, loop, loop_envoy, x):
        with pytest.warns(UserWarning, match="was never reached"):
            with loop_envoy.trace(x) as tracer:
                for step in tracer.iter[[0, 2, 7]]:
                    loop_envoy.block.output


class TestSourceIteration:
    """``tracer.iter`` over a `.source` op that loops within a single forward."""

    @pytest.fixture
    def op_envoy(self):
        torch.manual_seed(0)
        return Envoy(SourceOpLoop(inner=5))

    @pytest.fixture
    def y(self):
        torch.manual_seed(0)
        return torch.randn(1, 8)

    def test_all_invocations(self, op_envoy, y):
        with op_envoy.trace(y) as tracer:
            captured = nnsight.save([])
            for step in tracer.iter[:5]:
                captured.append(op_envoy.source.self_step_0.output)
        assert len(captured) == 5
        base = captured[0]
        for i, got in enumerate(captured):  # step(x, i) == x @ w + i
            assert torch.allclose(got, base + i)

    def test_specific_invocation(self, op_envoy, y):
        with op_envoy.trace(y) as tracer:
            captured = nnsight.save([])
            for step in tracer.iter[2]:
                captured.append(op_envoy.source.self_step_0.output)
        assert len(captured) == 1

    def test_sparse_invocations(self, op_envoy, y):
        with op_envoy.trace(y) as tracer:
            got = nnsight.save([])
            for step in tracer.iter[[0, 2, 4]]:
                got.append((step, op_envoy.source.self_step_0.output))
        assert [s for s, _ in got] == [0, 2, 4]
        base = got[0][1]
        for s, captured in got:
            assert torch.allclose(captured, base + s)

    def test_modify_specific_invocation(self, op_envoy, y):
        with op_envoy.trace(y) as tracer:
            baseline = nnsight.save(tracer.result)
        with op_envoy.trace(y) as tracer:
            for step in tracer.iter[2]:
                op_envoy.source.self_step_0.output[:] = 0
            modified = nnsight.save(tracer.result)
        assert not torch.allclose(baseline, modified)


class TestCache:
    def test_captures_all_outputs(self, envoy, model, x):
        with envoy.trace(x) as tracer:
            cache = tracer.cache()
        assert torch.allclose(cache["model.l1"].output, model.l1(x))
        assert cache["model.l1"].inputs is None  # outputs only by default

    def test_output_and_inputs_pair_up(self, envoy, x):
        with envoy.trace(x) as tracer:
            cache = tracer.cache(include_inputs=True)
        # l1's output is l2's input.
        assert torch.allclose(cache["model.l1"].output, cache["model.l2"].inputs[0][0])
        assert torch.allclose(cache["model.l2"].input, cache["model.l1"].output)

    def test_inputs_only(self, envoy, x):
        with envoy.trace(x) as tracer:
            cache = tracer.cache(include_inputs=True, include_output=False)
        assert cache["model.l1"].inputs is not None
        assert cache["model.l1"].output is None

    def test_subset_by_reference(self, envoy, x):
        with envoy.trace(x) as tracer:
            cache = tracer.cache(modules=[envoy.l2])
        assert cache.keys() == ["model.l2"]
        assert "model.l1" not in cache

    def test_navigation_matches_path(self, envoy, x):
        with envoy.trace(x) as tracer:
            cache = tracer.cache()
        # Both the full (cache.model.l1) and short (cache.l1) forms resolve.
        assert torch.equal(cache.model.l1.output, cache["model.l1"].output)
        assert torch.equal(cache.l1.output, cache["model.l1"].output)

    def test_cache_must_be_created_before_model_access(self, envoy, x):
        with pytest.raises(ValueError, match="before reading or modifying"):
            with envoy.trace(x) as tracer:
                envoy.l2.output = envoy.l2.output
                tracer.cache(modules=[envoy.l2])

    def test_cache_captures_skipped_output(self, envoy, x):
        with envoy.trace(x) as tracer:
            cache = tracer.cache()
            l1_out = envoy.l1.output.save()
            envoy.l2.skip(envoy.l1.output)  # l2 skipped, output replaced
        assert torch.equal(cache["model.l2"].output, l1_out)

    def test_captures_post_intervention(self, envoy, x):
        with envoy.trace(x) as tracer:
            cache = tracer.cache()
            envoy.l1.output = torch.zeros(2, 8)
        assert (cache["model.l1"].output == 0).all()

    def test_device_move(self, envoy, x):
        with envoy.trace(x) as tracer:
            cache = tracer.cache(device=torch.device("cpu"))
        assert cache["model.l1"].output.device == torch.device("cpu")

    def test_generation_accumulates_list(self):
        # A module visited every step of a loop is cached once per visit.
        torch.manual_seed(0)
        loop_envoy = Envoy(Loop(steps=5))
        x = torch.randn(2, 8)
        with loop_envoy.trace(x) as tracer:
            cache = tracer.cache(modules=[loop_envoy.block])
        assert len(cache["model.block"]) == 5
        outputs = cache["model.block"].output  # list, one per visit
        assert len(outputs) == 5


class OutputBlock(nn.Module):
    """A block with a submodule literally named `output` (BERT-style)."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.output = nn.Linear(8, 8)  # shadows Envoy.output

    def forward(self, x):
        return self.output(self.attn(x))


class Collide(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = OutputBlock()

    def forward(self, x):
        return self.block(x)


class TestOverloadedMount:
    @pytest.fixture
    def collide(self):
        torch.manual_seed(0)
        with pytest.warns(UserWarning, match="shadows"):
            return Envoy(Collide())

    def test_submodule_keeps_the_name(self, collide):
        assert isinstance(collide.block.output, Envoy)
        assert collide.block.output.path == "model.block.output"

    def test_submodule_is_navigable(self, collide):
        # weight lives on the real submodule, reachable through the shadowed name
        assert collide.block.output.weight.shape == (8, 8)

    def test_no_duplicate_children(self, collide):
        names = [c.path.rsplit(".", 1)[-1] for c in collide.block._children]
        assert sorted(names) == ["attn", "output"]

    def test_nnsight_proxy_relocated(self, collide):
        x = torch.randn(2, 8)
        ref = collide._module(x)
        with collide.trace(x):
            # the submodule's own output (== the block output here)
            sub = collide.block.output.output.save()
            # and the block's forward output via the relocated nns_output
            block_out = collide.block.nns_output.save()
        assert torch.allclose(sub, ref)
        assert torch.allclose(block_out, ref)

    def test_nns_setter_still_intervenes(self, collide):
        x = torch.randn(2, 8)
        with collide.trace(x):
            collide.block.nns_output = torch.zeros(2, 8)
            out = collide.output.save()
        assert (out == 0).all()


class TestSession:
    def test_cross_trace_intervention(self, envoy, x):
        with envoy.session():
            with envoy.trace(x):
                l1_out = envoy.l1.output.save()
            with envoy.trace(x):
                l1_out[:, 2] = 5
        assert (l1_out[:, 2] == 5).all()

    def test_conditional_trace(self, envoy, x):
        with envoy.session():
            if 5 > 0:
                with envoy.trace(x):
                    out = envoy.output.save()
        assert isinstance(out, torch.Tensor)

    def test_conditional_iteration(self, envoy, x):
        with envoy.session():
            result = nnsight.save([])
            for item in [0, 1, 2]:
                if item % 2 == 0:
                    with envoy.trace(x):
                        result.append(item)
        assert result == [0, 2]

    def test_loop_with_break_and_builtins(self, envoy):
        with envoy.session():
            l = nnsight.save([])
            l.append(0)
            for item in [1, 2, 3, 4]:
                if item == 3:
                    break
                l.append(item)
            l.append(5)
        assert l == [0, 1, 2, 5]

    def test_bridge_session_value_into_trace(self, envoy, x):
        with envoy.session():
            val = 3.0
            with envoy.trace(x):
                envoy.l1.output[:] = val
                l1_out = envoy.l1.output.save()
        assert torch.all(l1_out == 3.0)

    def test_value_flows_between_traces_without_save(self, envoy, x):
        # A session's defining feature: values pass between traces unsaved.
        with envoy.session():
            with envoy.trace(x):
                a = envoy.l1.output  # not saved
            with envoy.trace(x):
                b = (a + 1).save()  # `a` flowed in from the first trace
        assert b is not None

    def test_without_session_unsaved_value_is_dropped(self, envoy, x):
        with envoy.trace(x):
            a = envoy.l1.output  # not saved -> dropped at the outermost trace
        with pytest.raises(NameError):
            with envoy.trace(x):
                (a + 1).save()
