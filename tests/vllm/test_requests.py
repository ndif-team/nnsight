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
