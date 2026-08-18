"""Request tracking and lifecycle across the process boundary.

Each invoke becomes one vLLM request that the engine schedules on its own clock,
so the port has to keep three things straight: which tokens in a step's flat slab
belong to which request, which requests are still running, and which have finished
and can be forgotten. These tests exercise that bookkeeping directly — batched
requests of different lengths, generations that finish at different steps, and the
worker-side state left behind once a request is done.
"""

import pytest
import torch

pytest.importorskip("vllm")

import nnsight  # noqa: E402


def _worker_request_state(worker):
    """Read a worker's request bookkeeping, run there via ``collective_rpc``.

    The tracked workers live in the worker process; this returns how many there are
    and how many have been started (their greenlet exists), so the client can assert
    a finished request was cleaned up — collect drops the mediator, so both fall to 0.
    """
    requests = worker.model_runner.nnsight_requests
    return {
        "mediators": len(requests.mediators),
        "started": sum(
            1 for mediator in requests.mediators.values() if mediator.worker is not None
        ),
    }


def _request_state(model):
    return model.vllm_entrypoint.llm_engine.collective_rpc(_worker_request_state)[0]


class TestClientCleanup:
    @torch.no_grad()
    def test_mediators_cleared_after_trace(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            vllm_gpt2.logits.save()

        # The interleaver's per-trace workers are dropped when the trace ends.
        assert vllm_gpt2.interleaver.mediators == []

    @torch.no_grad()
    def test_repeated_traces_stay_correct(self, vllm_gpt2, ET_prompt):
        for _ in range(5):
            with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
                logits = vllm_gpt2.logits.save()
            assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"
            assert vllm_gpt2.interleaver.mediators == []


class TestWorkerCleanup:
    @torch.no_grad()
    def test_finished_request_is_forgotten(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            vllm_gpt2.logits.save()

        # collect_nnsight pops each finished request, so nothing is left behind.
        assert _request_state(vllm_gpt2)["mediators"] == 0

    @torch.no_grad()
    def test_generation_leaves_no_residue(self, vllm_gpt2, MSG_prompt):
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=5
        ) as tracer:
            logits = list().save()
            for _ in tracer.all():
                logits.append(vllm_gpt2.logits)

        assert len(logits) == 5
        state = _request_state(vllm_gpt2)
        assert state["mediators"] == 0
        assert state["started"] == 0


class TestBatchIsolation:
    @torch.no_grad()
    def test_clean_and_corrupted_are_independent(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                clean_hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
                clean_logits = vllm_gpt2.logits.save()

            with tracer.invoke(ET_prompt):
                out = vllm_gpt2.transformer.h[-2].mlp.output.clone()
                out[:] = 0
                vllm_gpt2.transformer.h[-2].mlp.output = out
                corrupted_hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
                corrupted_logits = vllm_gpt2.logits.save()

        assert not torch.all(clean_hs == 0)
        assert torch.all(corrupted_hs == 0)
        assert vllm_gpt2.tokenizer.decode(clean_logits.argmax(dim=-1)) == " Paris"
        assert vllm_gpt2.tokenizer.decode(corrupted_logits.argmax(dim=-1)) != " Paris"

    @torch.no_grad()
    def test_different_length_prompts_get_their_own_spans(
        self, vllm_gpt2, ET_prompt, MSG_prompt
    ):
        et_tokens = len(vllm_gpt2.tokenizer.encode(ET_prompt))
        msg_tokens = len(vllm_gpt2.tokenizer.encode(MSG_prompt))

        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                et_hs = vllm_gpt2.transformer.h[0].output.save()
                et_logits = vllm_gpt2.logits.save()

            with tracer.invoke(MSG_prompt):
                msg_hs = vllm_gpt2.transformer.h[0].output.save()
                msg_logits = vllm_gpt2.logits.save()

        # Each invoke is narrowed to exactly its own tokens within the shared slab,
        # not the whole batch.
        assert et_hs.shape[0] == et_tokens
        assert msg_hs.shape[0] == msg_tokens
        assert vllm_gpt2.tokenizer.decode(et_logits.argmax(dim=-1)) == " Paris"
        assert vllm_gpt2.tokenizer.decode(msg_logits.argmax(dim=-1)) == " New"

    @torch.no_grad()
    def test_identical_prompts_do_not_collide(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                first = vllm_gpt2.logits.save()
            with tracer.invoke(ET_prompt):
                second = vllm_gpt2.logits.save()

        # vLLM tags same-content requests with the same hash; the two saves must
        # still come home to their own invoke rather than overwriting each other.
        assert first is not second
        assert vllm_gpt2.tokenizer.decode(first.argmax(dim=-1)) == " Paris"
        assert vllm_gpt2.tokenizer.decode(second.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_three_invokes_stay_distinct(self, vllm_gpt2, ET_prompt, MSG_prompt):
        JP_prompt = "The capital of Japan is the city of"

        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                a = vllm_gpt2.logits.save()
            with tracer.invoke(MSG_prompt):
                b = vllm_gpt2.logits.save()
            with tracer.invoke(JP_prompt):
                c = vllm_gpt2.logits.save()

        # Three requests in one batch, each collected under its own invoke.
        assert a is not b and b is not c and a is not c
        assert a.shape[-1] == vllm_gpt2.tokenizer.vocab_size
        assert vllm_gpt2.tokenizer.decode(a.argmax(dim=-1)) == " Paris"
        assert vllm_gpt2.tokenizer.decode(b.argmax(dim=-1)) == " New"

    @torch.no_grad()
    def test_invoke_body_may_reference_module_globals(self, vllm_gpt2, ET_prompt):
        # An invoke body is serialized into its request; a module global it uses
        # (here `torch`) must travel with it, not be dropped by the reducer.
        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                vllm_gpt2.transformer.h[-2].mlp.output = torch.zeros_like(
                    vllm_gpt2.transformer.h[-2].mlp.output
                )
                logits = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) != " Paris"

    @torch.no_grad()
    def test_empty_invoke_contributes_no_request(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke():
                pass
            with tracer.invoke(ET_prompt):
                logits = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"


class TestConcurrentGeneration:
    @torch.no_grad()
    def test_invokes_finish_at_their_own_step_counts(
        self, vllm_gpt2, ET_prompt, MSG_prompt
    ):
        with vllm_gpt2.trace(temperature=0.0, top_p=1.0) as tracer:
            with tracer.invoke(ET_prompt, max_tokens=6):
                et_logits = list().save()
                for _ in tracer.iter[:6]:
                    et_logits.append(vllm_gpt2.logits)

            with tracer.invoke(MSG_prompt, max_tokens=3):
                msg_logits = list().save()
                for _ in tracer.iter[:3]:
                    msg_logits.append(vllm_gpt2.logits)

        # The shorter request finishes at step 3 while the longer runs to 6; the
        # batch is repacked each decode step and each invoke collects its own count.
        assert len(et_logits) == 6
        assert len(msg_logits) == 3
        assert _request_state(vllm_gpt2)["mediators"] == 0

    @torch.no_grad()
    def test_intervention_on_one_of_several_invokes(self, vllm_gpt2, MSG_prompt):
        # The same prompt runs twice side by side; only the second is intervened
        # on, and only its prediction moves — the corruption stays in its own request.
        with vllm_gpt2.trace(temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
            with tracer.invoke(MSG_prompt):
                clean = list().save()
                for _ in tracer.iter[:3]:
                    clean.append(vllm_gpt2.logits.argmax(dim=-1))

            with tracer.invoke(MSG_prompt):
                corrupted = list().save()
                for _ in tracer.iter[:3]:
                    out = vllm_gpt2.transformer.h[-2].output.clone()
                    out[:] = 0
                    vllm_gpt2.transformer.h[-2].output = out
                    corrupted.append(vllm_gpt2.logits.argmax(dim=-1))

        assert vllm_gpt2.tokenizer.batch_decode(clean) == [" New", " York", " City"]
        assert vllm_gpt2.tokenizer.batch_decode(corrupted) != [" New", " York", " City"]

    @torch.no_grad()
    def test_many_traces_leave_worker_empty(self, vllm_gpt2, ET_prompt, MSG_prompt):
        for prompt in (ET_prompt, MSG_prompt) * 3:
            with vllm_gpt2.trace(prompt, temperature=0.0, top_p=1, max_tokens=2):
                vllm_gpt2.logits.save()
        # No request bookkeeping accumulates across many traces.
        state = _request_state(vllm_gpt2)
        assert state["mediators"] == 0
        assert state["started"] == 0

    @torch.no_grad()
    def test_each_invoke_collects_its_own_generation(
        self, vllm_gpt2, ET_prompt, MSG_prompt
    ):
        # Each invoke keeps its own saved list, filled across its own generation
        # steps, and both come home from the one trace.
        with vllm_gpt2.trace(temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
            with tracer.invoke(ET_prompt):
                et = list().save()
                for _ in tracer.all():
                    et.append(vllm_gpt2.logits.argmax(dim=-1))

            with tracer.invoke(MSG_prompt):
                msg = list().save()
                for _ in tracer.all():
                    msg.append(vllm_gpt2.logits.argmax(dim=-1))

        assert len(et) == 3
        assert len(msg) == 3
        assert vllm_gpt2.tokenizer.batch_decode(msg) == [" New", " York", " City"]


class TestSeveralSequences:
    """``n > 1``: one prompt, several sampled continuations.

    vLLM fans the request into a child per sequence, each of which runs its own
    copy of the block against its own rows. So there are `n` of every saved
    value, and they come back as a list — one entry per sequence, in order.
    """

    @torch.no_grad()
    def test_saves_come_back_one_per_sequence(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, max_tokens=3, temperature=1.0, seed=0, n=2):
            hidden = vllm_gpt2.transformer.h[5].output.save()

        assert isinstance(hidden, list) and len(hidden) == 2
        prompt_rows = len(vllm_gpt2.tokenizer.encode(ET_prompt))
        for value in hidden:
            assert value.shape[0] == prompt_rows

    @torch.no_grad()
    def test_one_sequence_is_still_one_value(self, vllm_gpt2, ET_prompt):
        # The list is what "more than one" looks like; n=1 is unchanged.
        with vllm_gpt2.trace(ET_prompt, max_tokens=3, temperature=0.0):
            hidden = vllm_gpt2.transformer.h[5].output.save()

        assert isinstance(hidden, torch.Tensor)

    @torch.no_grad()
    def test_the_result_is_one_object_not_one_per_sequence(
        self, vllm_gpt2, ET_prompt
    ):
        # A request has one output carrying n completions, and every sequence's
        # block is served that same object — so it is not n values of anything.
        with vllm_gpt2.trace(
            ET_prompt, max_tokens=3, temperature=1.0, seed=0, n=2
        ) as tracer:
            result = tracer.result.save()

        assert type(result).__name__ == "RequestOutput"
        assert len(result.outputs) == 2

    @torch.no_grad()
    def test_each_completion_carries_its_own(self, vllm_gpt2_uncached, ET_prompt):
        # An installed block runs once per sequence too, and its values ride the
        # completion they belong to — next to that sequence's text and token ids.
        # This is the shape an async or serve caller reads, where there is no
        # variable to push a list into.
        model = vllm_gpt2_uncached

        with model.edit() as (tracer, edit):
            hidden = model.transformer.h[5].output.save()
        try:
            output = model.generate([ET_prompt], max_tokens=3, temperature=1.0,
                                    seed=0, n=2)[0]

            assert len(output.outputs) == 2
            prompt_rows = len(model.tokenizer.encode(ET_prompt))
            for completion in output.outputs:
                assert completion.saves["hidden"].shape[0] == prompt_rows
            # Each sequence's own object, not one shared between them.
            assert (
                output.outputs[0].saves["hidden"]
                is not output.outputs[1].saves["hidden"]
            )
        finally:
            edit.clear()

    @torch.no_grad()
    def test_the_sequences_diverge_where_they_should(self, vllm_gpt2, ET_prompt):
        # Same prompt, so the prefill matches; the sampled steps are each
        # sequence's own.
        with vllm_gpt2.trace(
            ET_prompt, max_tokens=4, temperature=1.0, seed=0, n=2
        ) as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[:4]:
                steps.append(vllm_gpt2.samples)
            result = tracer.result.save()

        assert isinstance(steps, list) and len(steps) == 2
        first, second = (
            [int(step.flatten()[0]) for step in sequence] for sequence in steps
        )
        assert first == list(result.outputs[0].token_ids)
        assert second == list(result.outputs[1].token_ids)

    @torch.no_grad()
    def test_no_worker_is_left_behind(self, vllm_gpt2, ET_prompt):
        # The child requests are named "{index}_{parent}"; when that went
        # unmatched nothing was collected and nothing was ever freed.
        engine = vllm_gpt2.vllm_entrypoint.llm_engine

        for _ in range(3):
            with vllm_gpt2.trace(ET_prompt, max_tokens=2, temperature=1.0,
                                 seed=0, n=2):
                hidden = vllm_gpt2.transformer.h[5].output.save()

        assert engine.collective_rpc("nnsight_request_count") == [0]


class TestSameNameAcrossInvokes:
    """Several invokes saving one name: separate values, not copies of one."""

    @torch.no_grad()
    def test_each_invoke_keeps_its_own(self, vllm_gpt2, ET_prompt, MSG_prompt):
        # This is the shape every steering/patching recipe has. Merging them kept
        # only the last, so three prompts came back holding one prompt's value.
        with vllm_gpt2.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(ET_prompt):
                hidden = vllm_gpt2.transformer.h[5].output.save()
            with tracer.invoke(MSG_prompt):
                hidden = vllm_gpt2.transformer.h[5].output.save()

        assert isinstance(hidden, list) and len(hidden) == 2
        # In invoke order, each sized to its own prompt.
        assert hidden[0].shape[0] == len(vllm_gpt2.tokenizer.encode(ET_prompt))
        assert hidden[1].shape[0] == len(vllm_gpt2.tokenizer.encode(MSG_prompt))

    @torch.no_grad()
    def test_distinct_names_are_untouched(self, vllm_gpt2, ET_prompt, MSG_prompt):
        with vllm_gpt2.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(ET_prompt):
                et = vllm_gpt2.transformer.h[5].output.save()
            with tracer.invoke(MSG_prompt):
                msg = vllm_gpt2.transformer.h[5].output.save()

        assert isinstance(et, torch.Tensor) and isinstance(msg, torch.Tensor)
        assert et.shape[0] != msg.shape[0]

    @torch.no_grad()
    def test_a_container_saved_above_the_invokes_still_merges(
        self, vllm_gpt2, ET_prompt, MSG_prompt
    ):
        # One object locally, so its per-request copies merge slot-wise rather
        # than coming back as a list of copies.
        with vllm_gpt2.trace(temperature=0.0, max_tokens=1) as tracer:
            rows = nnsight.save([None, None])
            with tracer.invoke(ET_prompt):
                rows[0] = vllm_gpt2.transformer.h[5].output
            with tracer.invoke(MSG_prompt):
                rows[1] = vllm_gpt2.transformer.h[5].output

        assert len(rows) == 2
        assert rows[0] is not None and rows[1] is not None
        assert rows[0].shape[0] == len(vllm_gpt2.tokenizer.encode(ET_prompt))
        assert rows[1].shape[0] == len(vllm_gpt2.tokenizer.encode(MSG_prompt))


class TestPerInvokeSampling:
    """An invoke's own sampling settings survive the trace-level ones.

    Trace-level settings fill in for whatever an invoke did not name. Working out
    "did not name" by comparing against a fresh `SamplingParams` cannot tell the
    value a caller passed from the one it would have had anyway — so an invoke
    asking for a setting that happens to be vLLM's default had it silently
    replaced, which is most of the values anyone types.
    """

    @torch.no_grad()
    def test_a_default_valued_setting_is_still_the_invokes(self, vllm_gpt2,
                                                           ET_prompt, MSG_prompt):
        # 16 is vLLM's own default max_tokens, and 3 is not; both are this
        # invoke's business, and the trace-level 4 applies to neither.
        with vllm_gpt2.trace(temperature=0.0, max_tokens=4) as tracer:
            with tracer.invoke(ET_prompt, max_tokens=16, ignore_eos=True):
                long = tracer.result.save()
            with tracer.invoke(MSG_prompt, max_tokens=3, ignore_eos=True):
                short = tracer.result.save()

        assert len(long.outputs[0].token_ids) == 16
        assert len(short.outputs[0].token_ids) == 3

    @torch.no_grad()
    def test_the_trace_level_setting_still_fills_in(self, vllm_gpt2, ET_prompt,
                                                    MSG_prompt):
        with vllm_gpt2.trace(temperature=0.0, max_tokens=4, ignore_eos=True) as tracer:
            with tracer.invoke(ET_prompt):
                first = tracer.result.save()
            with tracer.invoke(MSG_prompt, max_tokens=2):
                second = tracer.result.save()

        assert len(first.outputs[0].token_ids) == 4
        assert len(second.outputs[0].token_ids) == 2

    @torch.no_grad()
    def test_a_default_valued_temperature_is_kept(self, vllm_gpt2, ET_prompt):
        # temperature=1.0 is the default, and asking for it against a greedy
        # trace has to mean sampling. Two draws at temperature 1 diverge; two
        # greedy ones cannot.
        texts = set()
        for seed in (0, 1, 2, 3):
            with vllm_gpt2.trace(temperature=0.0, max_tokens=6) as tracer:
                with tracer.invoke(ET_prompt, temperature=1.0, seed=seed,
                                   ignore_eos=True):
                    result = tracer.result.save()
            texts.add(result.outputs[0].text)

        assert len(texts) > 1, f"every draw came back identical: {texts}"
