"""Blocks registered on the engine, which run for every request it handles.

A trace rides one request; a registration is installed once and applies to
everything the engine runs afterwards, including requests nnsight never created.
What it saves comes back on that request's ``RequestOutput``, by the same collect
a traced value uses, and is dropped as it goes.
"""

import pytest
import torch

pytest.importorskip("vllm")

import nnsight  # noqa: E402


def sampling(max_tokens: int = 3):
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)


class TestRegistration:
    @torch.no_grad()
    def test_untraced_requests_carry_their_own_values(self, vllm_gpt2_uncached):
        model = vllm_gpt2_uncached
        prompts = ["The Eiffel Tower is in", "Hello world", "A"]

        with model.edit() as (tracer, registration):
            hidden = model.transformer.h[5].output.save()
        try:
            outputs = model.generate(prompts, max_tokens=3, temperature=0.0,
                                     ignore_eos=True)

            assert len(outputs) == len(prompts)
            for output in outputs:
                # Its own value, sized to its own prompt.
                assert "hidden" in output.saves
                assert output.saves["hidden"].shape[0] == len(
                    output.prompt_token_ids
                )
            # Not one object shared between them.
            assert len({id(o.saves["hidden"]) for o in outputs}) == len(outputs)
        finally:
            registration.clear()

    @torch.no_grad()
    def test_clear_stops_it(self, vllm_gpt2_uncached, ET_prompt):
        model = vllm_gpt2_uncached

        with model.edit() as (tracer, registration):
            hidden = model.transformer.h[5].output.save()
        registration.clear()

        outputs = model.generate([ET_prompt], max_tokens=2, temperature=0.0,
                                 ignore_eos=True)
        assert "hidden" not in getattr(outputs[0], "saves", {})

    @torch.no_grad()
    def test_several_registrations_keep_their_own_names(self, vllm_gpt2_uncached,
                                                        ET_prompt):
        model = vllm_gpt2_uncached

        with model.edit() as (tracer, deep_reg):
            deep = model.transformer.h[8].output.save()
        with model.edit() as (tracer, shallow_reg):
            shallow = model.transformer.h[2].output.save()
        try:
            saves = model.generate([ET_prompt], max_tokens=2, temperature=0.0,
                                   ignore_eos=True)[0].saves
            # Both ran on the one request, each with a scope of its own.
            assert {"deep", "shallow"} <= set(saves)
            assert saves["deep"].shape == saves["shallow"].shape
        finally:
            deep_reg.clear()
            shallow_reg.clear()

    @torch.no_grad()
    def test_iteration_follows_the_generated_tokens(self, vllm_gpt2_uncached,
                                                    ET_prompt):
        # Without the tracer's iter/all a registered block finishes inside the
        # prefill and never sees a generated token.
        model = vllm_gpt2_uncached
        steps = 4

        with model.edit() as (tracer, registration):
            readout = nnsight.save([])
            for step in tracer.iter[:steps]:
                readout.append(model.transformer.h[5].output[-1])
        try:
            outputs = model.generate([ET_prompt], max_tokens=steps,
                                     temperature=0.0, ignore_eos=True)
            assert len(outputs[0].saves["readout"]) == steps
        finally:
            registration.clear()

    @torch.no_grad()
    def test_edits_every_request(self, vllm_gpt2_uncached, ET_prompt):
        model = vllm_gpt2_uncached
        before = model.generate([ET_prompt], max_tokens=3, temperature=0.0,
                                ignore_eos=True)[0].outputs[0].text

        with model.edit() as (tracer, registration):
            model.transformer.h[5].output[:] = 0
        try:
            during = model.generate([ET_prompt], max_tokens=3, temperature=0.0,
                                    ignore_eos=True)[0].outputs[0].text
        finally:
            registration.clear()
        after = model.generate([ET_prompt], max_tokens=3, temperature=0.0,
                               ignore_eos=True)[0].outputs[0].text

        assert during != before
        assert after == before

    @torch.no_grad()
    def test_a_trace_reaches_its_registered_value_through_result(
        self, vllm_gpt2_uncached, ET_prompt
    ):
        model = vllm_gpt2_uncached

        with model.edit() as (tracer, registration):
            hidden = model.transformer.h[5].output.save()
        try:
            with model.trace(ET_prompt, temperature=0.0, max_tokens=2,
                             ignore_eos=True) as tracer:
                own = model.transformer.h[2].output.save()
                result = tracer.result.save()

            # The trace's own value arrives as a variable; the registration's
            # rides the output it is handed.
            assert own is not None
            assert "hidden" in result.saves
            # What the block was handed is the snapshot taken at collect time, so
            # it carries the registered value and the generation but not the
            # trace's own saves — those are what that collect is computing, and
            # one of them is this object.
            assert "own" not in result.saves
            assert result.outputs[0].token_ids
        finally:
            registration.clear()


class TestEditHandles:
    @torch.no_grad()
    def test_clear_edits_clears_every_one(self, vllm_gpt2_uncached, ET_prompt):
        model = vllm_gpt2_uncached

        with model.edit() as (tracer, deep):
            deep_hidden = model.transformer.h[8].output.save()
        with model.edit() as (tracer, shallow):
            shallow_hidden = model.transformer.h[2].output.save()

        model.clear_edits()

        assert deep.cleared and shallow.cleared
        assert model._installed_edits == []
        saves = getattr(
            model.generate([ET_prompt], max_tokens=2, temperature=0.0,
                           ignore_eos=True)[0],
            "saves",
            {},
        )
        assert "deep_hidden" not in saves and "shallow_hidden" not in saves

    @torch.no_grad()
    def test_a_cleared_edit_drops_out_of_the_list(self, vllm_gpt2_uncached):
        model = vllm_gpt2_uncached

        with model.edit() as (tracer, edit):
            hidden = model.transformer.h[5].output.save()
        assert model._installed_edits == [edit]

        edit.clear()
        assert model._installed_edits == []
        # Clearing twice is a no-op, not an error.
        edit.clear()

    def test_inplace_false_is_refused(self, vllm_gpt2_uncached):
        # There is no copy to edit instead — the engine is what every caller shares.
        with pytest.raises(ValueError, match="inplace"):
            vllm_gpt2_uncached.edit(inplace=False)


class TestBarrierIsRefused:
    """Each invoke is its own request, so the blocks never meet — say so."""

    def test_in_a_trace(self, vllm_gpt2):
        with pytest.raises(NotImplementedError, match="barrier"):
            with vllm_gpt2.trace(temperature=0.0, max_tokens=1) as tracer:
                barrier = tracer.barrier(2)

    def test_in_an_edit(self, vllm_gpt2_uncached):
        # Not entered: an edit's body runs per request on the worker, so calling
        # it there would only surface as that request's deferred error.
        tracer = vllm_gpt2_uncached.edit()
        with pytest.raises(NotImplementedError, match="barrier"):
            tracer.barrier(2)


class TestGenerateWithoutABlock:
    @torch.no_grad()
    def test_returns_request_outputs(self, vllm_gpt2, ET_prompt):
        outputs = vllm_gpt2.generate([ET_prompt], max_tokens=3, temperature=0.0,
                                     ignore_eos=True)

        assert isinstance(outputs, list) and len(outputs) == 1
        assert outputs[0].outputs[0].text
        assert len(outputs[0].outputs[0].token_ids) == 3

    @torch.no_grad()
    def test_still_traces_inside_a_with_block(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.generate(ET_prompt, temperature=0.0, max_tokens=1) as tracer:
            logits = vllm_gpt2.logits.save()

        assert logits.shape[-1] == vllm_gpt2.tokenizer.vocab_size

    @torch.no_grad()
    def test_max_new_tokens_is_accepted(self, vllm_gpt2, ET_prompt):
        outputs = vllm_gpt2.generate([ET_prompt], max_new_tokens=2,
                                     temperature=0.0, ignore_eos=True)
        assert len(outputs[0].outputs[0].token_ids) == 2


class TestResult:
    @torch.no_grad()
    def test_result_is_the_finished_request_output(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, max_tokens=3,
                             ignore_eos=True) as tracer:
            result = tracer.result.save()

        assert type(result).__name__ == "RequestOutput"
        assert len(result.outputs[0].token_ids) == 3

    @torch.no_grad()
    def test_each_invoke_gets_its_own(self, vllm_gpt2, ET_prompt, MSG_prompt):
        with vllm_gpt2.trace(temperature=0.0, max_tokens=2,
                             ignore_eos=True) as tracer:
            with tracer.invoke(ET_prompt):
                first = tracer.result.save()
            with tracer.invoke(MSG_prompt):
                second = tracer.result.save()

        assert first.request_id != second.request_id
        assert first.prompt != second.prompt
