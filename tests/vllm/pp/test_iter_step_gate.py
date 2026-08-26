"""The step gate: open-ended iteration paced by the driver, single process.

An open-ended ``tracer.iter`` loop has no termination condition of its own;
the model ending is what stops it, through a dangling park. That exit only
exists if each step parks. A body whose reads are all remote-owned (pipeline
parallelism) or that performs no reads at all never parks, and before the
gate it spun the thread forever. Now such a step parks on ``STEP_GATE``,
which the driver serves once per generation step, restoring the designed
exit.
"""

import warnings

import pytest

from nnsight.intervention.interleaver import STEP_GATE, Interleaver, Mediator
from nnsight.intervention.iterator import Iterations


def make_mediator(interleaver, block, glbls):
    glbls = {"Mediator": Mediator, **glbls}
    mediator = Mediator(compile(block, "<step-gate-test>", "exec"), glbls, {})
    interleaver.mediators.append(mediator)
    return mediator


def test_park_free_open_ended_loop_is_paced_and_terminated():
    interleaver = Interleaver()
    mediator = make_mediator(
        interleaver,
        """
out = []
for s in STEPS:
    out.append(s)
""",
        {"STEPS": Iterations(0, None)},
    )
    with interleaver:
        # The worker parks on the gate after step 0's body; each serve
        # releases exactly one more step.
        for _ in range(3):
            interleaver.handle(STEP_GATE, None)
    # Generation over with the worker parked on the gate: the dangling check
    # (which drivers run after the model finishes) unwinds the loop quietly —
    # the loop simply ended with the run.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        interleaver.check_dangling_mediators()
    assert mediator.lcls["out"] == [0, 1, 2, 3]
    assert not mediator.alive


def test_bounded_loop_needs_no_gate():
    interleaver = Interleaver()
    mediator = make_mediator(
        interleaver,
        """
out = []
for s in STEPS:
    out.append(s)
""",
        {"STEPS": Iterations(0, 4)},
    )
    with interleaver:
        pass  # a bounded park-free loop runs to its bound and finishes
    assert mediator.lcls["out"] == [0, 1, 2, 3]
    assert not mediator.alive


def test_pin_relaxing_body_still_advances_one_step_per_serve():
    # A remote read under PP is served in place by the intercept, which may
    # relax the iteration pin without parking. The gate park pins its own
    # tag (the serves already seen, at least one past the previous gate
    # park), so each serve releases exactly one step regardless of what the
    # body did to the pin.
    class Intercepting(Interleaver):
        def intercept(self, mediator, event, location, rest):
            if location == "model.remote.output":
                if mediator.iteration:
                    mediator.iteration = None
                return ("remote-value",)
            return None

    interleaver = Intercepting()
    mediator = make_mediator(
        interleaver,
        """
seen = []
for s in STEPS:
    seen.append(Mediator.value("model.remote.output"))
""",
        {"STEPS": Iterations(0, None)},
    )
    with interleaver:
        for _ in range(3):
            interleaver.handle(STEP_GATE, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        interleaver.check_dangling_mediators()
    assert mediator.lcls["seen"] == ["remote-value"] * 4
    assert not mediator.alive


def test_pre_loop_park_then_open_loop_rides_later_serves():
    # A park before the loop (a save of a value served after the forward)
    # lets gate serves pass while the worker waits elsewhere. Each gate park
    # pins to the serves already seen, so the loop rides the remaining
    # steps' serves.
    interleaver = Interleaver()
    mediator = make_mediator(
        interleaver,
        """
first = Mediator.value("logits")
out = []
for s in STEPS:
    out.append(s)
""",
        {"STEPS": Iterations(0, None)},
    )
    with interleaver:
        interleaver.handle(STEP_GATE, None)  # step 0 boundary; parked on logits
        interleaver.handle("logits", "L0")
        interleaver.handle(STEP_GATE, None)  # step 1 boundary
        interleaver.handle(STEP_GATE, None)  # step 2 boundary
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        interleaver.check_dangling_mediators()
    assert mediator.lcls["first"] == "L0"
    assert mediator.lcls["out"] == [0, 1, 2]
    assert not mediator.alive


def test_parking_body_never_touches_the_gate():
    interleaver = Interleaver()
    mediator = make_mediator(
        interleaver,
        """
values = []
for s in STEPS:
    values.append(Mediator.value("model.h.0.output"))
""",
        {"STEPS": Iterations(0, None)},
    )
    with interleaver:
        interleaver.handle("model.h.0.output", 10)
        interleaver.handle("model.h.0.output", 11)
    with pytest.warns(UserWarning):
        interleaver.check_dangling_mediators()
    # Each step parked on the read itself; the loop was paced by the served
    # location and ended by the dangling read, with no gate involvement.
    assert mediator.lcls["values"] == [10, 11]
    assert not mediator.alive


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
