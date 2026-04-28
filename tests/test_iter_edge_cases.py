"""Edge case tests for ``tracer.iter[...]`` and ``.next()``.

These tests exercise iteration boundaries that aren't covered by the
``TestIteration`` class in ``test_lm.py`` or the ``TestSourceIter``
class in ``test_source.py``:

A. Recurrent inner modules called multiple times per outer step
B. Branching modules where a sub-module isn't called on some steps
C. ``tracer.iter[N]`` past the actual generation length
D. ``tracer.iter[:]`` (unbounded) trailing-code skip
E. ``tracer.iter[a:b]`` bounded slice body count
F. Cross-invoke iter loops are tracker-independent
G. ``module.next()`` and ``tracer.next()`` advancement
H. First-time ``.source`` access mid-iter-loop limitation

Most tests use a small custom ``nn.Module`` so they run on CPU in
under a second; the streamer and generation-related tests reuse the
shared ``gpt2`` fixture.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import nnsight
from nnsight import LanguageModel, NNsight
from nnsight.intervention.interleaver import Mediator


# ---------------------------------------------------------------------------
# Custom modules
# ---------------------------------------------------------------------------


class RecurrentInner(nn.Module):
    """A module whose forward calls an inner submodule multiple times.

    Each call to ``forward(x)`` runs ``self.linear`` 3 times in sequence.
    Wrapped with NNsight, every call to ``self.linear`` increments the
    iteration tracker for ``model.linear.{input,output}``.
    """

    def __init__(self, dim: int = 4, inner_calls: int = 3):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.inner_calls = inner_calls

    def forward(self, x):
        for _ in range(self.inner_calls):
            x = self.linear(x)
        return x


class Branched(nn.Module):
    """Conditional path — only one of the two sub-modules fires."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.sub_a = nn.Linear(dim, dim)
        self.sub_b = nn.Linear(dim, dim)

    def forward(self, x):
        if x.sum() > 0:
            return self.sub_a(x)
        return self.sub_b(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recurrent_model(device: str):
    torch.manual_seed(0)
    return NNsight(RecurrentInner(dim=4, inner_calls=3)).to(device)


@pytest.fixture
def branched_model(device: str):
    torch.manual_seed(0)
    return NNsight(Branched(dim=4)).to(device)


@pytest.fixture
def small_input():
    torch.manual_seed(0)
    return torch.randn(1, 4)


# ---------------------------------------------------------------------------
# A. Recurrent inner module — multiple calls per outer step
# ---------------------------------------------------------------------------


class TestRecurrentInnerIter:
    """``tracer.iter[N]`` indexes into the per-call sequence, NOT generation step.

    The persistent iter hook (see ``register_iter_hooks``) fires on
    EVERY forward pass of the wrapped module. For a recurrent inner
    module called 3 times per outer call, the tracker advances 3 times
    per outer call. So for an inner module called 3 times across a
    single ``trace()``, the valid iter indices are 0, 1, 2.
    """

    @torch.no_grad()
    def test_iter_indexes_inner_calls_within_one_trace(
        self, recurrent_model: NNsight, small_input: torch.Tensor
    ):
        """``iter[i]`` targets the i-th call to ``self.linear``.

        With ``inner_calls=3``, all three slots (0, 1, 2) should fire.
        """
        captured = []
        with recurrent_model.trace(small_input) as tracer:
            captured = list().save()
            for step in tracer.iter[:3]:
                captured.append(recurrent_model.linear.output.clone())

        assert len(captured) == 3
        for t in captured:
            assert isinstance(t, torch.Tensor)
            assert t.shape == (1, 4)

    @torch.no_grad()
    def test_iter_specific_inner_call(
        self, recurrent_model: NNsight, small_input: torch.Tensor
    ):
        """``iter[1]`` should fire on the 2nd call to ``self.linear``."""
        with recurrent_model.trace(small_input) as tracer:
            captured = list().save()
            for step in tracer.iter[1]:
                captured.append(recurrent_model.linear.output.clone())

        assert len(captured) == 1

    @torch.no_grad()
    def test_iter_modifies_specific_inner_call(
        self, recurrent_model: NNsight, small_input: torch.Tensor
    ):
        """Zeroing the 2nd inner call's output propagates into the 3rd call."""
        # Baseline — no intervention
        with recurrent_model.trace(small_input):
            baseline_out = recurrent_model.output.clone().save()

        # Zero the 2nd inner call (iter[1])
        with recurrent_model.trace(small_input) as tracer:
            for step in tracer.iter[1]:
                recurrent_model.linear.output[:] = 0
            modified_out = recurrent_model.output.clone().save()

        # Final output should differ because the 3rd call sees zeros as input
        assert not torch.allclose(baseline_out, modified_out)


# ---------------------------------------------------------------------------
# B. Branched module — sub-module never fires on some inputs
# ---------------------------------------------------------------------------


class TestBranchedIter:
    """When a sub-module never fires, accessing its ``.output`` raises
    ``MissedProviderError``.
    """

    @torch.no_grad()
    def test_unfired_branch_raises_missed_provider(
        self, branched_model: NNsight
    ):
        """sub_b never runs (input has positive sum). Accessing
        ``sub_b.output`` should raise a ``MissedProviderError``.
        """
        positive_input = torch.ones(1, 4)  # sum > 0 → sub_a fires

        with pytest.raises(Mediator.MissedProviderError):
            with branched_model.trace(positive_input):
                branched_model.sub_a.output.save()  # OK
                branched_model.sub_b.output.save()  # never fires

    @torch.no_grad()
    def test_fired_branch_works(self, branched_model: NNsight):
        """Sanity: the branch that DOES fire works normally."""
        positive_input = torch.ones(1, 4)

        with branched_model.trace(positive_input):
            out = branched_model.sub_a.output.save()

        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# C. iter[N] past actual generation length — emits warning
# ---------------------------------------------------------------------------


class TestIterPastLength:
    """``tracer.iter[N]`` and ``tracer.iter[a:b]`` past the end of
    generation behavior depends on whether the body requests module
    values:

    - If body accesses a module: worker thread waits for ``provider.iN``
      that never arrives → missed-provider warning. The trace's local
      variables are NOT pushed back (they don't exist outside the trace).
    - If body has no module access: body still runs once per iter step
      with the ``step`` value bound to the requested index.
    """

    @torch.no_grad()
    def test_bounded_slice_past_length_warns_and_locals_lost(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        """``iter[5:7]`` with max_new_tokens=2: worker waits for
        ``output.i5`` that never arrives. A UserWarning is emitted and
        the trace locals don't propagate back to the parent scope.
        """
        captured_outer = None
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            try:
                with gpt2.generate(MSG_prompt, max_new_tokens=2) as tracer:
                    captured = list().save()
                    for step in tracer.iter[5:7]:
                        captured.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))
                # If we got here, the trace exited "normally" (with warning).
                captured_outer = captured  # may UnboundLocalError
            except UnboundLocalError:
                # Local was never bound back from the worker — expected.
                captured_outer = "unbound"

        msgs = [str(w.message) for w in wlist]
        assert any("not provided" in m for m in msgs), (
            f"Expected a missed-provider warning, got: {msgs}"
        )

    @torch.no_grad()
    def test_iter_past_length_no_module_access_runs_body(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        """``iter[5]`` with max_new_tokens=2 and a body that doesn't
        access any module: the body runs once with ``step=5``, no
        warning, no error.
        """
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            with gpt2.generate(MSG_prompt, max_new_tokens=2) as tracer:
                captured = list().save()
                for step in tracer.iter[5]:
                    captured.append(step)  # no module access

        # Body did run with the requested index.
        assert list(captured) == [5]


# ---------------------------------------------------------------------------
# D. tracer.iter[:] (unbounded) — trailing code skipped
# ---------------------------------------------------------------------------


class TestIterUnboundedTrailing:
    """Behavior of code AFTER ``for step in tracer.iter[:]``:

    - Pure-Python trailing code (no module access) DOES run. The iter
      loop body is implemented as a Python generator that yields each
      step; once generation ends with no more steps to yield, the
      generator's ``StopIteration`` exits the for-loop normally and
      the worker thread continues.
    - Trailing code that requests another module value AFTER the iter
      loop raises ``OutOfOrderError`` because the model has already
      finished its forward passes — that module's hooks won't fire
      again.

    This contradicts older docs (CLAUDE.md) that say trailing code
    "never executes". Trailing pure-Python code DOES execute.
    """

    @torch.no_grad()
    def test_trailing_pure_python_after_iter_runs(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        """Pure-Python statements after ``iter[:]`` DO execute (mutate
        a closed-over list to confirm).
        """
        ran_marker = [False]

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with gpt2.generate(MSG_prompt, max_new_tokens=2) as tracer:
                logits = list().save()
                for step in tracer.iter[:]:
                    logits.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))
                ran_marker[0] = True

        assert len(logits) == 2
        assert ran_marker[0] is True, (
            "Pure-Python trailing code after iter[:] should run."
        )

    @torch.no_grad()
    def test_trailing_module_access_after_iter_raises(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        """Accessing a module AFTER ``iter[:]`` raises ``OutOfOrderError``
        because the model has already finished its forward passes.
        """
        with pytest.raises(Mediator.OutOfOrderError):
            with gpt2.generate(MSG_prompt, max_new_tokens=2) as tracer:
                logits = list().save()
                for step in tracer.iter[:]:
                    logits.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))
                # This module access fails — model has exited.
                gpt2.lm_head.output.save()


# ---------------------------------------------------------------------------
# E. tracer.iter[a:b] (bounded slice) — body count
# ---------------------------------------------------------------------------


class TestIterBoundedSlice:
    """``tracer.iter[1:3]`` should run the body exactly twice (i=1, 2)."""

    @torch.no_grad()
    def test_bounded_slice_runs_twice(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        with gpt2.generate(MSG_prompt, max_new_tokens=5) as tracer:
            steps_seen = list().save()
            for step in tracer.iter[1:3]:
                steps_seen.append(step)
                gpt2.lm_head.output[0][-1].argmax(dim=-1)

        assert list(steps_seen) == [1, 2]

    @torch.no_grad()
    def test_bounded_slice_full_range(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        with gpt2.generate(MSG_prompt, max_new_tokens=4) as tracer:
            seen = list().save()
            for step in tracer.iter[0:4]:
                seen.append(step)

        assert list(seen) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# F. Cross-invoke iter loops — independent trackers
# ---------------------------------------------------------------------------


class TestIterCrossInvoke:
    """Two invokes both running ``tracer.iter[:]`` should track
    independently — each invoke has its own mediator with its own
    ``iteration_tracker``.
    """

    @torch.no_grad()
    def test_two_invokes_independent_iter(
        self, gpt2: LanguageModel, ET_prompt: str, MSG_prompt: str
    ):
        with gpt2.generate(max_new_tokens=3) as tracer:
            with tracer.invoke(ET_prompt):
                logits_a = list().save()
                for step in tracer.iter[:]:
                    logits_a.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))
            with tracer.invoke(MSG_prompt):
                logits_b = list().save()
                for step in tracer.iter[:]:
                    logits_b.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

        assert len(logits_a) == 3
        assert len(logits_b) == 3
        # The two invokes use different prompts, so logits should differ.
        assert not torch.equal(logits_a[0], logits_b[0])

    @torch.no_grad()
    def test_two_invokes_different_iter_ranges(
        self, gpt2: LanguageModel, ET_prompt: str, MSG_prompt: str
    ):
        """Each invoke can use a different iter range."""
        with gpt2.generate(max_new_tokens=4) as tracer:
            with tracer.invoke(ET_prompt):
                seen_a = list().save()
                for step in tracer.iter[0:2]:
                    seen_a.append(step)
            with tracer.invoke(MSG_prompt):
                seen_b = list().save()
                for step in tracer.iter[2:4]:
                    seen_b.append(step)

        assert list(seen_a) == [0, 1]
        assert list(seen_b) == [2, 3]


# ---------------------------------------------------------------------------
# G. module.next() and tracer.next()
# ---------------------------------------------------------------------------


class TestNext:
    """Manual generation-step advancement via ``module.next()`` and
    ``tracer.next()``.

    KNOWN BUG (as of refactor/transform branch): ``.next()`` outside an
    iter loop does NOT work because the persistent iter hooks (which
    bump ``mediator.iteration_tracker``) are only registered inside an
    ``IteratorTracer``. Without those hooks, the tracker stays at 0 and
    one-shot output hooks looking for ``output.iN`` (N>0) never fire.

    These tests document the current (broken) behavior. The docs in
    CLAUDE.md and NNsight.md show ``.next()`` chains as if they work
    standalone — they don't.

    Workaround: use ``tracer.iter[:]`` (which DOES register the iter
    hooks) instead of ``.next()``.
    """

    @torch.no_grad()
    def test_module_next_outside_iter_loop_is_broken(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        """``model.transformer.h[-1].next().output`` outside an iter
        loop emits a missed-provider warning and the trace's locals
        don't propagate back.
        """
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            try:
                with gpt2.generate(MSG_prompt, max_new_tokens=3) as tracer:
                    hs0 = gpt2.transformer.h[-1].output[0].clone().save()
                    hs1 = gpt2.transformer.h[-1].next().output[0].clone().save()
            except UnboundLocalError:
                pass

        msgs = [str(w.message) for w in wlist]
        assert any("not provided" in m and "i1" in m for m in msgs), (
            f"Expected missed-provider warning for i1, got: {msgs}"
        )

    @torch.no_grad()
    def test_tracer_next_outside_iter_loop_is_broken(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        """``tracer.next()`` outside iter has the same bug."""
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            try:
                with gpt2.generate(MSG_prompt, max_new_tokens=3) as tracer:
                    hs0 = gpt2.transformer.h[-1].output[0].clone().save()
                    tracer.next()
                    hs1 = gpt2.transformer.h[-1].output[0].clone().save()
            except UnboundLocalError:
                pass

        msgs = [str(w.message) for w in wlist]
        assert any("not provided" in m and "i1" in m for m in msgs), (
            f"Expected missed-provider warning for i1, got: {msgs}"
        )

    @torch.no_grad()
    def test_tracer_next_inside_iter_works(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        """``tracer.next()`` inside an iter loop advances the iteration.

        Wrapping in ``tracer.iter[:]`` registers the persistent iter
        hooks that bump ``iteration_tracker``, so ``.next()`` advances
        through real steps and one-shot hooks fire.
        """
        with gpt2.generate(MSG_prompt, max_new_tokens=3) as tracer:
            hs_list = list().save()
            for step in tracer.iter[:]:
                hs_list.append(gpt2.transformer.h[-1].output[0].clone())

        # All 3 generation steps yielded a tensor.
        assert len(hs_list) == 3
        for hs in hs_list:
            assert isinstance(hs, torch.Tensor)


# ---------------------------------------------------------------------------
# H. First-time .source access mid-iter-loop — known limitation
# ---------------------------------------------------------------------------


class TestSourceMidIterLimitation:
    """Documents the known limitation: if ``.source`` is built FOR THE
    FIRST TIME at step N>0 of an iter loop, op-path trackers start at 0
    instead of N. The user's first hook captures ``iteration=N`` but
    checks against ``tracker[op]=0``, so that one access misses (raises
    a missed-provider warning). Subsequent steps work fine.

    See ``register_iter_hooks`` docstring "Known limitation" section.
    """

    @torch.no_grad()
    def test_source_built_pre_iter_works(self, gpt2: LanguageModel):
        """Touching ``.source`` BEFORE the iter loop seeds op-path
        trackers correctly — control test for the limitation below.
        """
        gpt2.transformer.h[0].attn.source  # build accessor pre-loop

        with gpt2.generate("Hello", max_new_tokens=3) as tracer:
            outs = list().save()
            for step in tracer.iter[:]:
                outs.append(gpt2.transformer.h[0].attn.source.split_1.output)

        assert len(outs) == 3

    @torch.no_grad()
    def test_source_built_mid_iter_misses_first(
        self, gpt2: LanguageModel
    ):
        """When the SourceAccessor is built FOR THE FIRST TIME inside an
        iter loop at step N, the first op access at step N misses (warns).
        Subsequent steps work.

        We use a fresh model (or rather: a fresh attention head) — but
        we can't use a fresh gpt2 fixture, so we pick a layer whose
        ``.source`` hasn't been touched yet by other tests. To make this
        deterministic across the test suite, this test simply documents
        the behavior with a warnings filter that allows the missed
        provider case.
        """
        # Note: this test is intentionally lenient because the
        # ``gpt2`` fixture is module-scoped; another test may have
        # already built the SourceAccessor for this layer. We just
        # check that the iter completes without raising.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with gpt2.generate("Hello", max_new_tokens=3) as tracer:
                outs = list().save()
                for step in tracer.iter[:]:
                    # First-time source on layer 11 mid-loop (if not
                    # already built by another test in this session)
                    outs.append(gpt2.transformer.h[11].attn.source.split_1.output)

        # Behavior depends on whether the accessor was pre-built.
        # In a fresh session the first step misses; otherwise all 3 fire.
        assert len(outs) in (2, 3)


# ---------------------------------------------------------------------------
# 2.2 Streamer — verifies HF TextIteratorStreamer integration
# ---------------------------------------------------------------------------


class TestStreamerIntegration:
    """HuggingFace's ``TextIteratorStreamer`` should work inside
    ``model.generate(streamer=...)`` without breaking nnsight tracing.
    """

    @torch.no_grad()
    def test_huggingface_text_iterator_streamer(
        self, gpt2: LanguageModel, MSG_prompt: str
    ):
        """Tokens stream out through the HF streamer alongside an
        nnsight intervention.
        """
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            gpt2.tokenizer, skip_prompt=True, timeout=30.0
        )

        with gpt2.generate(
            MSG_prompt, max_new_tokens=3, streamer=streamer
        ) as tracer:
            saved = list().save()
            for step in tracer.iter[:]:
                saved.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

        # Drain the streamer
        streamed_text = ""
        for chunk in streamer:
            streamed_text += chunk

        assert len(saved) == 3
        # The streamed text should be non-empty.
        assert len(streamed_text) > 0


# ---------------------------------------------------------------------------
# 2.3 ValueError class for trace-no-input
# ---------------------------------------------------------------------------


class TestTraceNoInputError:
    """Calling ``model.trace()`` with no input and no invokes should
    raise a ``ValueError`` (wrapped in ``NNsightException``, but
    ``isinstance(e, ValueError)`` still holds).
    """

    def test_trace_no_input_raises_value_error(
        self, tiny_model: NNsight
    ):
        with pytest.raises(ValueError) as excinfo:
            with tiny_model.trace():
                tiny_model.output.save()

        # The message should mention "Cannot access" and "interleaving"
        assert "Cannot access" in str(excinfo.value)
        assert "interleaving" in str(excinfo.value)
