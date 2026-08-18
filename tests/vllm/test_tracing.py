"""End-to-end tracing against a single-rank vLLM engine.

Covers the client-facing surface: reading logits and samples, multi-token
generation, sampling settings, activation reads and edits, and the several input
forms an invoke accepts. Interventions are written exactly as for a local model;
that they land here proves the whole round trip — serialized onto the request,
run in the worker, saved values shipped home.
"""

import pytest
import torch

pytest.importorskip("vllm")


class TestLogits:
    @torch.no_grad()
    def test_single_logit(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()

        assert logits.shape[-1] == vllm_gpt2.tokenizer.vocab_size
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_logits_edit_changes_prediction(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            vllm_gpt2.logits = torch.zeros_like(vllm_gpt2.logits)
            logits = vllm_gpt2.logits.save()

        assert torch.all(logits == 0)


class TestGeneration:
    @torch.no_grad()
    def test_multi_token_generation(self, vllm_gpt2, MSG_prompt):
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=3
        ) as tracer:
            logits = list().save()
            for _ in tracer.iter[0:3]:
                logits.append(vllm_gpt2.logits)

        assert vllm_gpt2.tokenizer.batch_decode(
            [logit.argmax(dim=-1) for logit in logits]
        ) == [" New", " York", " City"]

    @torch.no_grad()
    def test_max_token_count(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, max_tokens=10) as tracer:
            logits = list().save()
            for _ in tracer.all():
                logits.append(vllm_gpt2.logits)

        assert len(logits) == 10

    @torch.no_grad()
    def test_generate_alias(self, vllm_gpt2, MSG_prompt):
        with vllm_gpt2.generate(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=3
        ) as tracer:
            logits = list().save()
            for _ in tracer.iter[0:3]:
                logits.append(vllm_gpt2.logits)

        assert vllm_gpt2.tokenizer.batch_decode(
            [logit.argmax(dim=-1) for logit in logits]
        ) == [" New", " York", " City"]

    @torch.no_grad()
    def test_max_new_tokens_is_rewritten(self, vllm_gpt2, MSG_prompt):
        # max_new_tokens must become max_tokens: the request stops at 3, so a cache
        # watching every step sees exactly 3. (Asserting only iter[:3]==3 would pass
        # even if the rewrite were dropped and the default max_tokens ran.)
        with vllm_gpt2.generate(
            MSG_prompt, temperature=0.0, top_p=1.0, max_new_tokens=3
        ) as tracer:
            cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])
            for _ in tracer.all():
                vllm_gpt2.logits

        assert len(cache["model.transformer.h.0"]) == 3

    @torch.no_grad()
    def test_max_new_tokens_on_trace_too(self, vllm_gpt2, MSG_prompt):
        # trace() rewrites it as well, not just generate().
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_new_tokens=3
        ) as tracer:
            cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])
            for _ in tracer.all():
                vllm_gpt2.logits

        assert len(cache["model.transformer.h.0"]) == 3


class TestSampling:
    @torch.no_grad()
    def test_samples_are_the_drawn_ids(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
            samples = vllm_gpt2.samples.save()

        # Greedy: the sampled id is the argmax of the logits it was drawn from.
        assert samples.item() == logits.argmax(dim=-1).item()

    @torch.no_grad()
    def test_samples_edit_replaces_the_drawn_id(self, vllm_gpt2, ET_prompt):
        # Overwriting `samples` swaps the token the engine continues from; reading it
        # back in the same step returns the forced id, not the greedy draw.
        forced = vllm_gpt2.tokenizer.encode(" Chicago")[0]
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1, max_tokens=1):
            greedy = vllm_gpt2.logits.argmax(dim=-1).save()
            vllm_gpt2.samples = torch.full_like(vllm_gpt2.samples, forced)
            after = vllm_gpt2.samples.save()

        assert greedy.item() != forced          # the model would not have drawn it
        assert after.item() == forced            # but the swap took

    @torch.no_grad()
    def test_temperature_changes_samples(self, vllm_gpt2, MSG_prompt):
        with vllm_gpt2.trace(max_tokens=3) as tracer:
            with tracer.invoke(MSG_prompt, temperature=0.8, top_p=0.95, seed=1):
                sampled = list().save()
                for _ in tracer.iter[0:3]:
                    sampled.append(vllm_gpt2.samples.item())

            with tracer.invoke(MSG_prompt, temperature=0.0, top_p=1.0):
                greedy = list().save()
                for _ in tracer.iter[0:3]:
                    greedy.append(vllm_gpt2.samples.item())

        # One id at a time: these are plain ints, and `batch_decode` reads a flat
        # list of them as a sequence each in transformers 4 but as one sequence in
        # transformers 5, so it cannot state this across both. (The other decodes
        # in this suite pass tensors, which are unambiguous.)
        assert [
            vllm_gpt2.tokenizer.decode([token]) for token in greedy
        ] == [" New", " York", " City"]
        assert sampled != greedy

    @torch.no_grad()
    def test_seed_is_reproducible(self, vllm_gpt2, MSG_prompt):
        def run():
            with vllm_gpt2.trace(
                MSG_prompt, temperature=0.8, top_p=0.95, seed=123, max_tokens=3
            ) as tracer:
                drawn = list().save()
                for _ in tracer.iter[0:3]:
                    drawn.append(vllm_gpt2.samples.item())
            return drawn

        assert run() == run()


class TestInterventions:
    @torch.no_grad()
    def test_read_activation_spans_the_prompt(self, vllm_gpt2, ET_prompt):
        n_tokens = len(vllm_gpt2.tokenizer.encode(ET_prompt))

        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            hs = vllm_gpt2.transformer.h[-2].mlp.output.save()

        assert hs.shape[0] == n_tokens

    @torch.no_grad()
    def test_zero_activation_swap(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            vllm_gpt2.transformer.h[-2].mlp.output = torch.zeros_like(
                vllm_gpt2.transformer.h[-2].mlp.output
            )
            hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
            logits = vllm_gpt2.logits.save()

        assert torch.all(hs == 0)
        # Zeroing a late MLP moves the prediction off the clean answer.
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) != " Paris"

    @torch.no_grad()
    def test_in_place_edit(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            out = vllm_gpt2.transformer.h[-2].mlp.output.clone()
            out[:] = 0
            vllm_gpt2.transformer.h[-2].mlp.output = out
            hs = vllm_gpt2.transformer.h[-2].mlp.output.save()

        assert torch.all(hs == 0)

    @torch.no_grad()
    def test_read_module_input(self, vllm_gpt2, ET_prompt):
        n_tokens = len(vllm_gpt2.tokenizer.encode(ET_prompt))

        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            hs = vllm_gpt2.transformer.h[5].input.save()

        assert hs.shape[0] == n_tokens

    @torch.no_grad()
    def test_edit_module_input(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            vllm_gpt2.transformer.h[5].input = torch.zeros_like(
                vllm_gpt2.transformer.h[5].input
            )
            logits = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) != " Paris"

    @torch.no_grad()
    def test_multi_layer_intervention(self, vllm_gpt2, ET_prompt):
        # Editing an early layer and reading a later one in the same trace: the
        # later read is downstream of the edit, and the prediction reflects it.
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            vllm_gpt2.transformer.h[3].output = torch.zeros_like(
                vllm_gpt2.transformer.h[3].output
            )
            later = vllm_gpt2.transformer.h[8].output.save()
            logits = vllm_gpt2.logits.save()

        assert (later != 0).any()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) != " Paris"

    @torch.no_grad()
    def test_edit_at_one_generation_step(self, vllm_gpt2, MSG_prompt):
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=5
        ) as tracer:
            zeroed = list().save()
            for it in tracer.iter[:]:
                if it == 2:
                    out = vllm_gpt2.transformer.h[-2].output.clone()
                    out[0][:] = 0
                    vllm_gpt2.transformer.h[-2].output = out
                zeroed.append(torch.all(vllm_gpt2.transformer.h[-2].output[0] == 0))

        # Only the third step sees the zeroed activation.
        assert [bool(z) for z in zeroed] == [False, False, True, False, False]


class TestInputForms:
    @torch.no_grad()
    def test_token_id_list(self, vllm_gpt2, ET_prompt):
        token_ids = vllm_gpt2.tokenizer.encode(ET_prompt)

        with vllm_gpt2.trace(token_ids, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_tokenizer_output_dict(self, vllm_gpt2, ET_prompt):
        hf_output = vllm_gpt2.tokenizer(ET_prompt, return_tensors="pt")

        with vllm_gpt2.trace(dict(hf_output), temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_dict_without_attention_mask(self, vllm_gpt2, ET_prompt):
        token_ids = vllm_gpt2.tokenizer.encode(ET_prompt)

        with vllm_gpt2.trace({"input_ids": token_ids}, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_attention_mask_selects_tokens(self, vllm_gpt2, ET_prompt):
        token_ids = vllm_gpt2.tokenizer.encode(ET_prompt)
        # A leading zero in the mask drops the first token from the prompt.
        mask = [0] + [1] * (len(token_ids) - 1)

        with vllm_gpt2.trace(
            {"input_ids": token_ids, "attention_mask": mask}, temperature=0.0, top_p=1
        ):
            hs = vllm_gpt2.transformer.h[0].output.save()

        assert hs.shape[0] == len(token_ids) - 1

    @torch.no_grad()
    def test_dict_with_tensor_input_ids(self, vllm_gpt2, ET_prompt):
        # input_ids as a tensor rather than a list, no attention mask (issue #622).
        hf_output = vllm_gpt2.tokenizer(ET_prompt, return_tensors="pt")

        with vllm_gpt2.trace(
            {"input_ids": hf_output["input_ids"]}, temperature=0.0, top_p=1
        ):
            logits = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_mixed_string_and_token_invokes(self, vllm_gpt2, ET_prompt, MSG_prompt):
        # One invoke takes token ids, the other a string; each resolves to its own
        # request and prediction.
        et_tokens = vllm_gpt2.tokenizer.encode(ET_prompt)

        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(et_tokens):
                et_logits = vllm_gpt2.logits.save()
            with tracer.invoke(MSG_prompt):
                msg_logits = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(et_logits.argmax(dim=-1)) == " Paris"
        assert vllm_gpt2.tokenizer.decode(msg_logits.argmax(dim=-1)) == " New"

    @torch.no_grad()
    def test_rejects_multiple_prompts_per_invoke(self, vllm_gpt2, ET_prompt, MSG_prompt):
        with pytest.raises(ValueError):
            with vllm_gpt2.trace([ET_prompt, MSG_prompt], temperature=0.0):
                vllm_gpt2.logits.save()

    @torch.no_grad()
    def test_empty_invoke_with_interventions_raises(self, vllm_gpt2, ET_prompt):
        # An empty invoke has no vLLM request; interventions in its body would be
        # silently dropped, so it is refused.
        with pytest.raises(ValueError, match="no prompt"):
            with vllm_gpt2.trace(temperature=0.0) as tracer:
                with tracer.invoke():
                    vllm_gpt2.transformer.h[0].output = torch.zeros_like(
                        vllm_gpt2.transformer.h[0].output
                    )
                with tracer.invoke(ET_prompt):
                    vllm_gpt2.logits.save()

    @torch.no_grad()
    def test_unknown_trace_sampling_kwarg_raises(self, vllm_gpt2, ET_prompt):
        # A typo'd trace-level sampling setting is refused, not silently ignored.
        with pytest.raises(TypeError, match="temperatur"):
            with vllm_gpt2.trace(temperatur=0.0) as tracer:
                with tracer.invoke(ET_prompt):
                    vllm_gpt2.logits.save()


class TestEarlyStop:
    @torch.no_grad()
    def test_stop_halts_capture(self, vllm_gpt2, MSG_prompt):
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1, max_tokens=5
        ) as tracer:
            steps = list().save()
            for it in tracer.iter[:5]:
                steps.append(vllm_gpt2.logits.argmax(dim=-1))
                if it == 1:
                    tracer.stop()

        # Capture halts at the stop; the two steps up to it are kept.
        assert len(steps) == 2
        assert vllm_gpt2.tokenizer.batch_decode(steps) == [" New", " York"]

    @torch.no_grad()
    def test_stop_finishes_the_request(self, vllm_gpt2, MSG_prompt):
        # The engine stops scheduling the request, so far fewer than max_tokens
        # steps run — counted by a cache that watches every step until it ends.
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1, max_tokens=20
        ) as tracer:
            cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])
            for it in tracer.iter[:20]:
                vllm_gpt2.logits
                if it == 1:
                    tracer.stop()

        assert len(cache["model.transformer.h.0"]) < 20

    @torch.no_grad()
    def test_stop_finishes_the_request_under_min_tokens(self, vllm_gpt2, MSG_prompt):
        # min_tokens defers the scheduler's stop check, so a single forced
        # end-of-sequence would not retire the request. Re-forcing it every step ends
        # the request once min_tokens is met — still well short of max_tokens.
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1, max_tokens=40, min_tokens=4
        ) as tracer:
            cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])
            for it in tracer.iter[:40]:
                vllm_gpt2.logits
                if it == 0:
                    tracer.stop()

        steps = len(cache["model.transformer.h.0"])
        assert 4 <= steps < 40

    @torch.no_grad()
    def test_stop_under_ignore_eos_runs_to_max_tokens(self, vllm_gpt2, MSG_prompt):
        # ignore_eos is the residual: forcing EOS can't retire the request, so it runs
        # the full max_tokens. The stop is control flow, so no error surfaces and the
        # engine stays healthy.
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1, max_tokens=6, ignore_eos=True
        ) as tracer:
            cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])
            for it in tracer.iter[:6]:
                vllm_gpt2.logits
                if it == 1:
                    tracer.stop()

        assert len(cache["model.transformer.h.0"]) == 6
        with vllm_gpt2.trace(MSG_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " New"

    @torch.no_grad()
    def test_engine_healthy_after_stop(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(
            ET_prompt, temperature=0.0, top_p=1, max_tokens=5
        ) as tracer:
            for it in tracer.iter[:5]:
                vllm_gpt2.logits
                if it == 0:
                    tracer.stop()

        # A later trace still runs on the same engine.
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"


class TestDeferredErrors:
    @torch.no_grad()
    def test_error_surfaces_to_client(self, vllm_gpt2, ET_prompt):
        # An error in intervention code comes back to the trace that wrote it,
        # naming the original exception.
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
                _ = 1 / 0
                vllm_gpt2.logits.save()

    @torch.no_grad()
    def test_barrier_across_invokes_surfaces(self, vllm_gpt2, ET_prompt, MSG_prompt):
        # A barrier coordinates blocks within one process; across independent vLLM
        # requests each gets its own copy, so the count is never reached and it
        # surfaces as an error when the run ends rather than hanging forever.
        with pytest.raises(RuntimeError, match="barrier"):
            with vllm_gpt2.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
                barrier = tracer.barrier(2)
                with tracer.invoke(ET_prompt):
                    barrier()
                    vllm_gpt2.logits.save()
                with tracer.invoke(MSG_prompt):
                    barrier()
                    vllm_gpt2.logits.save()

    @torch.no_grad()
    def test_engine_survives_error(self, vllm_gpt2, ET_prompt):
        try:
            with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
                _ = 1 / 0
                vllm_gpt2.logits.save()
        except RuntimeError:
            pass

        # The error ended only its own request; a later trace runs normally.
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_out_of_order_read_surfaces(self, vllm_gpt2, ET_prompt):
        # Reading a layer the forward already ran past leaves the worker parked with
        # nowhere to be served. When the request finishes it is surfaced as an
        # OutOfOrderError rather than left to hang silently to max_tokens.
        with pytest.raises(RuntimeError, match="already ran past"):
            with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1, max_tokens=3):
                _later = vllm_gpt2.transformer.h[5].output
                _earlier = vllm_gpt2.transformer.h[0].output
                vllm_gpt2.logits.save()

    @torch.no_grad()
    def test_out_of_order_edit_surfaces(self, vllm_gpt2, ET_prompt):
        # Same as the read case but the parked worker is mid-edit (a swap), whose
        # pending event carries the replacement value — a differently shaped tuple.
        # It must still surface as an OutOfOrderError, not crash the shared engine.
        with pytest.raises(RuntimeError, match="already ran past"):
            with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1, max_tokens=3):
                later = vllm_gpt2.transformer.h[5].output
                vllm_gpt2.transformer.h[3].output = later
                vllm_gpt2.logits.save()

        # The engine is unharmed — a later trace runs normally.
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_error_in_one_invoke_isolated(self, vllm_gpt2, ET_prompt, MSG_prompt):
        with pytest.raises(RuntimeError):
            with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
                with tracer.invoke(ET_prompt):
                    vllm_gpt2.logits.save()
                with tracer.invoke(MSG_prompt):
                    _ = 1 / 0
                    vllm_gpt2.logits.save()

        # The engine and its other tenants are unharmed.
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"


class TestCache:
    @torch.no_grad()
    def test_cache_specific_modules(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(
            ET_prompt, temperature=0.0, top_p=1, max_tokens=1
        ) as tracer:
            cache = tracer.cache(
                modules=[vllm_gpt2.transformer.h[0], vllm_gpt2.transformer.h[-1]]
            )

        assert "model.transformer.h.0" in cache
        assert "model.transformer.h.11" in cache
        assert "model.transformer.h.5" not in cache
        assert cache["model.transformer.h.0"].output is not None

    @torch.no_grad()
    def test_cache_all_modules(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(
            ET_prompt, temperature=0.0, top_p=1, max_tokens=1
        ) as tracer:
            cache = tracer.cache()

        assert cache["model.transformer.h.0"].output is not None
        assert cache["model.transformer.h.11"].output is not None

    @torch.no_grad()
    def test_cache_excludes_inputs_by_default(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(
            ET_prompt, temperature=0.0, top_p=1, max_tokens=1
        ) as tracer:
            cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])

        assert cache["model.transformer.h.0"].output is not None
        assert cache["model.transformer.h.0"].inputs is None

    @torch.no_grad()
    def test_cache_includes_inputs_when_asked(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(
            ET_prompt, temperature=0.0, top_p=1, max_tokens=1
        ) as tracer:
            cache = tracer.cache(
                modules=[vllm_gpt2.transformer.h[0]], include_inputs=True
            )

        assert cache["model.transformer.h.0"].inputs is not None

    @torch.no_grad()
    def test_cache_narrows_to_each_invoke(self, vllm_gpt2, ET_prompt, MSG_prompt):
        et_n = len(vllm_gpt2.tokenizer.encode(ET_prompt))
        msg_n = len(vllm_gpt2.tokenizer.encode(MSG_prompt))

        with vllm_gpt2.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
            with tracer.invoke(ET_prompt):
                et_cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])
            with tracer.invoke(MSG_prompt):
                msg_cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])

        # Each invoke's cache records only its own tokens, not the combined batch.
        assert et_cache["model.transformer.h.0"].output.shape[0] == et_n
        assert msg_cache["model.transformer.h.0"].output.shape[0] == msg_n

    @torch.no_grad()
    def test_cache_captures_each_generation_step(self, vllm_gpt2, MSG_prompt):
        n_prompt = len(vllm_gpt2.tokenizer.encode(MSG_prompt))

        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1, max_tokens=3
        ) as tracer:
            cache = tracer.cache(modules=[vllm_gpt2.transformer.h[0]])

        entry = cache["model.transformer.h.0"]
        # One capture per step over the whole request, not just the prefill.
        assert len(entry) == 3
        outputs = entry.output
        # The prefill spans every prompt token, then one token per decode step.
        assert outputs[0].shape[0] == n_prompt
        assert outputs[1].shape[0] == 1
        assert outputs[2].shape[0] == 1


class TestLazyDispatch:
    """Building without ``dispatch=True``, then tracing.

    The path a user takes when they construct a model and only later run
    something on it: `__init__` builds the meta tree, and the first trace calls
    `dispatch`, which brings up the engine and re-points the tree at it. Every
    other test here dispatches eagerly, so nothing else covers it.
    """

    @torch.no_grad()
    def test_first_trace_dispatches_and_reads(self, ET_prompt):
        from nnsight.modeling.vllm import VLLM

        model = VLLM("gpt2", gpu_memory_utilization=0.1)
        assert not model.dispatched
        # The meta tree is there to write interventions against before any engine.
        assert model.tokenizer is not None
        assert model.transformer.h[5] is not None

        with model.trace(ET_prompt, temperature=0.0, top_p=1):
            hidden = model.transformer.h[5].output.save()
            logits = model.logits.save()

        assert model.dispatched
        assert hidden.shape[0] == len(model.tokenizer.encode(ET_prompt))
        assert model.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_dispatch_does_not_rebuild_the_meta_tree(self, ET_prompt):
        # `__init__` already built one, and dispatch is about to re-point the envoy
        # tree at whatever `_load` returns — so building a second, identical tree
        # is a whole architecture instantiation for nothing.
        from unittest import mock

        from nnsight.modeling.vllm import VLLM

        model = VLLM("gpt2", gpu_memory_utilization=0.1)
        tree = model._module

        with mock.patch.object(
            model, "_load_meta", side_effect=AssertionError("rebuilt the meta tree")
        ):
            model.dispatch()

        assert model.dispatched
        assert model._module is tree

    @torch.no_grad()
    def test_an_edit_written_before_dispatch_still_lands(self, ET_prompt):
        # `edit()` dispatches for itself, since the block has to go to an engine.
        from nnsight.modeling.vllm import VLLM

        model = VLLM("gpt2", gpu_memory_utilization=0.1,
                     enable_prefix_caching=False)
        with model.edit() as (tracer, edit):
            hidden = model.transformer.h[5].output.save()
        try:
            assert model.dispatched
            output = model.generate([ET_prompt], max_tokens=2, temperature=0.0,
                                    ignore_eos=True)[0]
            assert output.saves["hidden"].shape[0] == len(output.prompt_token_ids)
        finally:
            edit.clear()


class TestPromptForms:
    """What one invoke will take as its prompt.

    vLLM's own prompt dicts are `TypedDict`s — plain dicts at runtime — so they
    used to be taken for a tokenizer's output and die on `KeyError: 'input_ids'`.
    """

    @torch.no_grad()
    def test_a_string(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_a_token_id_list(self, vllm_gpt2, ET_prompt):
        ids = vllm_gpt2.tokenizer.encode(ET_prompt)
        with vllm_gpt2.trace(ids, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_a_tokenizer_output(self, vllm_gpt2, ET_prompt):
        inputs = vllm_gpt2.tokenizer(ET_prompt)
        with vllm_gpt2.trace(inputs, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_a_vllm_tokens_prompt(self, vllm_gpt2, ET_prompt):
        from vllm import TokensPrompt

        ids = vllm_gpt2.tokenizer.encode(ET_prompt)
        with vllm_gpt2.trace(
            TokensPrompt(prompt_token_ids=ids), temperature=0.0, top_p=1
        ):
            hidden = vllm_gpt2.transformer.h[5].output.save()
            logits = vllm_gpt2.logits.save()

        assert hidden.shape[0] == len(ids)
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_a_vllm_text_prompt(self, vllm_gpt2, ET_prompt):
        with vllm_gpt2.trace({"prompt": ET_prompt}, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.save()
        assert vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_each_invoke_takes_its_own_form(self, vllm_gpt2, ET_prompt, MSG_prompt):
        # The forms mix freely, since each invoke is converted on its own.
        from vllm import TokensPrompt

        ids = vllm_gpt2.tokenizer.encode(MSG_prompt)
        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                et = vllm_gpt2.logits.save()
            with tracer.invoke(TokensPrompt(prompt_token_ids=ids)):
                msg = vllm_gpt2.logits.save()

        assert vllm_gpt2.tokenizer.decode(et.argmax(dim=-1)) == " Paris"
        assert vllm_gpt2.tokenizer.decode(msg.argmax(dim=-1)) == " New"


class TestModelRunnerSelection:
    """nnsight instruments one of vLLM's two GPU model runners, and says so."""

    def test_it_asks_for_the_runner_it_instruments(self):
        # Set before the engine is built, since the worker processes inherit it.
        import os

        from nnsight.modeling.vllm import VLLM

        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        VLLM._require_v1_model_runner()
        assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"

    def test_asking_for_the_other_one_is_refused(self):
        # Overriding it silently would be worse: the engine comes up with no
        # instrumentation and the first collect fails with a missing method.
        import os

        from nnsight.modeling.vllm import VLLM

        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        try:
            with pytest.raises(NotImplementedError, match="V2 GPU model runner"):
                VLLM._require_v1_model_runner()
        finally:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
