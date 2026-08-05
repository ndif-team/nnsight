"""Language-model-level tests on a real ``TransformersModel`` (gpt2).

Covers generation, activation read/modify, input setting, source ops, iteration
over generation, early stop, gradients, caching, sessions, and module aliasing.
"""

import importlib.util

import pytest
import torch

import nnsight
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty
from nnsight.modeling.transformers import TransformersModel

PROMPT = "Madison Square Garden is located in the city of"


class Heads(Envoy):
    """A custom Envoy exposing a per-head ``.heads`` view via an eproperty: the
    preprocess reshapes ``[B, S, H]`` into ``[B, n_heads, S, head_dim]`` and the
    transform writes an edited view back into the module's real layout."""

    n_heads = 12

    @eproperty(key="output")
    def heads(self, value):
        b, s, h = value.shape
        return value.view(b, s, self.n_heads, h // self.n_heads).transpose(1, 2)

    @heads.transform
    def heads(self, value):
        b, nh, s, hd = value.shape
        return value.transpose(1, 2).reshape(b, s, nh * hd)


@pytest.fixture(scope="module")
def gpt2():
    return TransformersModel("gpt2", task="text-generation", dispatch=True)


@pytest.fixture(scope="module")
def lm():
    from nnsight import LanguageModel

    return LanguageModel("gpt2", dispatch=True)


@pytest.fixture(scope="module")
def llama():
    # A modern decoder architecture (RoPE / GQA / RMSNorm) — a different envoy tree
    # (model.layers) and a bare-tensor layer output, unlike gpt2.
    return TransformersModel(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        task="text-generation",
        dispatch=True,
    )


class TestLanguageModel:
    """The deprecated ``LanguageModel`` still behaves as it did before the rewrite:
    token ids out of ``generate`` (read them off ``tracer.result``), decoded a step
    at a time through ``model.generator.streamer.output``, and greedy unless asked
    otherwise."""

    @torch.no_grad()
    def test_generate_returns_token_ids(self, lm):
        # Where TransformersModel yields the pipeline's decoded records, this yields
        # the prompt and completion as ids: 9 prompt tokens + 3 generated.
        with lm.generate(PROMPT, max_new_tokens=3) as tracer:
            out = tracer.result.save()
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 12)
        assert lm.tokenizer.decode(out[0]) == (
            "Madison Square Garden is located in the city of New York City"
        )

    @torch.no_grad()
    def test_generation_is_greedy_by_default(self, lm):
        # The checkpoint's task_specific_params ask the text-generation pipeline for
        # do_sample=True; generating through the model doesn't, so neither does this.
        with lm.generate(PROMPT, max_new_tokens=3) as tracer:
            first = tracer.result.save()
        with lm.generate(PROMPT, max_new_tokens=3) as tracer:
            second = tracer.result.save()
        assert torch.equal(first, second)

    @torch.no_grad()
    def test_generator_output_is_the_result(self, lm):
        # Two ways to the same value: the module the output is passed through, and
        # the return of the call itself.
        with lm.generate(PROMPT, max_new_tokens=3) as tracer:
            through_generator = lm.generator.output.save()
            returned = tracer.result.save()
        assert torch.equal(through_generator, returned)

    @torch.no_grad()
    def test_streamer_gives_a_step_at_a_time(self, lm):
        # The prompt arrives as one block, then a token per step.
        with lm.generate(PROMPT, max_new_tokens=3) as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[:3]:
                steps.append(lm.generator.streamer.output)
        assert len(steps) == 3
        assert steps[0].shape[-1] > 1  # the prompt
        assert all(step.shape[-1] == 1 for step in steps[1:])  # one new token each

    @torch.no_grad()
    def test_intervention_changes_generation(self, lm):
        # Zeroing the embeddings changes what comes out — the trace body is running
        # against the real generate, not a copy of it.
        with lm.generate(PROMPT, max_new_tokens=3) as tracer:
            clean = tracer.result.save()
        with lm.generate(PROMPT, max_new_tokens=3) as tracer:
            lm.transformer.wte.output = lm.transformer.wte.output * 0
            zeroed = tracer.result.save()
        assert not torch.equal(clean, zeroed)

    @torch.no_grad()
    def test_batched_generate_stacks_prompts(self, lm):
        with lm.generate([PROMPT, "The Eiffel Tower is in"], max_new_tokens=3) as tracer:
            out = tracer.result.save()
        assert out.shape[0] == 2
        assert lm.tokenizer.decode(out[0]).startswith("Madison Square Garden")

    @torch.no_grad()
    def test_invokes_generate_per_prompt(self, lm):
        # The ids are a batched tensor, so each invoke reads its own rows out of
        # them — the same way it reads its own rows of an activation.
        with lm.generate(max_new_tokens=3) as tracer:
            with tracer.invoke(PROMPT):
                first = lm.transformer.h[0].output.save()
                first_ids = tracer.result.save()
            with tracer.invoke("The Eiffel Tower is in"):
                second = lm.transformer.h[0].output.save()
                second_ids = tracer.result.save()
        assert first.shape[0] == 1 and second.shape[0] == 1
        assert first_ids.shape[0] == 1 and second_ids.shape[0] == 1
        # The shorter prompt is left-padded up to the longer one, so read past the
        # padding to see what each invoke actually generated from.
        decode = lambda ids: lm.tokenizer.decode(ids[0], skip_special_tokens=True)
        assert decode(first_ids).startswith("Madison Square Garden")
        assert decode(second_ids).startswith("The Eiffel Tower")

    def test_tokenizer_kwargs_reach_the_tokenizer(self):
        from nnsight import LanguageModel

        model = LanguageModel("gpt2", tokenizer_kwargs={"padding_side": "right"})
        assert model.tokenizer.padding_side == "right"

    def test_task_is_text_generation(self, lm):
        assert lm.task == "text-generation"

    @torch.no_grad()
    def test_sampling_can_be_asked_for(self, lm):
        # Greedy is only the default; a generate kwarg still wins over it.
        torch.manual_seed(0)
        with lm.generate(PROMPT, max_new_tokens=5, do_sample=True, top_k=50) as tracer:
            first = tracer.result.save()
        torch.manual_seed(1)
        with lm.generate(PROMPT, max_new_tokens=5, do_sample=True, top_k=50) as tracer:
            second = tracer.result.save()
        assert not torch.equal(first, second)

    @torch.no_grad()
    def test_caller_generation_config_wins(self, lm):
        from transformers import GenerationConfig

        with lm.generate(PROMPT, generation_config=GenerationConfig(max_new_tokens=2)) as tracer:
            out = tracer.result.save()
        assert out.shape == (1, 11)  # 9 prompt tokens + 2

    @torch.no_grad()
    def test_num_return_sequences_stacks(self, lm):
        # Every returned sequence becomes a row, not just the first.
        kwargs = dict(max_new_tokens=3, num_return_sequences=2, do_sample=True)
        with lm.generate(PROMPT, **kwargs) as tracer:
            out = tracer.result.save()
        assert out.shape[0] == 2

    @torch.no_grad()
    def test_called_directly_without_a_block(self, lm):
        # No `with`: generate just generates, and still returns ids.
        out = lm.generate(PROMPT, max_new_tokens=3)
        assert isinstance(out, torch.Tensor) and out.shape == (1, 12)

    @torch.no_grad()
    def test_generate_takes_token_ids(self, lm):
        # Generating goes through the model, so it takes the inputs a forward takes
        # — not only the text a pipeline would tokenize for itself.
        ids = lm.tokenizer(PROMPT).input_ids
        with lm.generate(ids, max_new_tokens=3) as tracer:
            out = tracer.result.save()
        assert out.shape == (1, 12)
        assert lm.tokenizer.decode(out[0]).endswith("New York City")

    @torch.no_grad()
    def test_generate_takes_a_tensor(self, lm):
        ids = lm.tokenizer(PROMPT, return_tensors="pt").input_ids
        with lm.generate(ids, max_new_tokens=3) as tracer:
            out = tracer.result.save()
        assert out.shape == (1, 12)

    @torch.no_grad()
    def test_generate_takes_an_encoding(self, lm):
        enc = lm.tokenizer(PROMPT, return_tensors="pt")
        with lm.generate(enc, max_new_tokens=3) as tracer:
            out = tracer.result.save()
        assert out.shape == (1, 12)


class TestGeneration:
    @torch.no_grad()
    def test_generate_returns_token_ids(self, gpt2):
        # generate goes through the model: the result is the prompt plus completion
        # as ids (9 prompt tokens + 3 generated), not the pipeline's decoded records.
        with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            out = tracer.result.save()
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 12)
        assert gpt2.tokenizer.decode(out[0]) == (
            "Madison Square Garden is located in the city of New York City"
        )

    @torch.no_grad()
    def test_generate_and_pipe_differ(self, gpt2):
        # The two entry points on the same model: generate yields token ids off the
        # model, pipe yields the task pipeline's decoded records.
        with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            ids = tracer.result.save()
        with gpt2.pipe(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            records = tracer.result.save()
        assert isinstance(ids, torch.Tensor)
        assert isinstance(records, list) and "generated_text" in records[0]

    @torch.no_grad()
    def test_generation_is_greedy_by_default(self, gpt2):
        # The checkpoint's task_specific_params ask the text-generation pipeline for
        # do_sample=True; generating through the model doesn't, so neither does this.
        with gpt2.generate(PROMPT, max_new_tokens=3) as tracer:
            first = tracer.result.save()
        with gpt2.generate(PROMPT, max_new_tokens=3) as tracer:
            second = tracer.result.save()
        assert torch.equal(first, second)

    @torch.no_grad()
    def test_generator_output_matches_result(self, gpt2):
        # The deprecated model.generator.output reads the same ids tracer.result does.
        with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            through_generator = gpt2.generator.output.save()
            returned = tracer.result.save()
        assert torch.equal(through_generator, returned)

    @torch.no_grad()
    def test_greedy_generation_text(self, gpt2):
        with gpt2.pipe(PROMPT, max_new_tokens=5, do_sample=False) as tracer:
            result = tracer.result.save()
        assert (
            result[0]["generated_text"]
            == "Madison Square Garden is located in the city of New York City.\n"
        )

    @torch.no_grad()
    def test_save_hidden_states_and_input(self, gpt2):
        with gpt2.trace(PROMPT):
            hs_in = gpt2.transformer.h[-1].input.save()
            hs_out = gpt2.transformer.h[-1].output.save()
        assert hs_in.shape[-1] == 768  # (batch, seq, hidden)
        assert hs_out.shape[-1] == 768


class TestCombinedWith:
    """A trace combined with another context manager on the same ``with`` line —
    ``with torch.no_grad(), model.trace(...):`` — captures and interleaves the block
    the same as a trace on its own line, whichever order the items come in."""

    def test_no_grad_then_trace(self, gpt2):
        with torch.no_grad(), gpt2.trace(PROMPT):
            hidden = gpt2.transformer.h[0].output[0].save()
        assert hidden.shape[-1] == 768

    def test_trace_then_no_grad(self, gpt2):
        with gpt2.trace(PROMPT), torch.no_grad():
            hidden = gpt2.transformer.h[0].output[0].save()
        assert hidden.shape[-1] == 768

    def test_combined_with_edit_lands(self, gpt2):
        with torch.no_grad(), gpt2.trace(PROMPT):
            clean = gpt2.output.logits[0, -1].argmax().save()
        with torch.no_grad(), gpt2.trace(PROMPT):
            gpt2.transformer.h[-1].output[0][:] = 0
            edited = gpt2.output.logits[0, -1].argmax().save()
        assert clean.item() != edited.item()

    def test_combined_with_binds_tracer(self, gpt2):
        # `as tracer` binds through the multi-item header, so tracer.result reads.
        with torch.no_grad(), gpt2.generate(
            PROMPT, max_new_tokens=3, do_sample=False
        ) as tracer:
            ids = tracer.result.save()
        assert isinstance(ids, torch.Tensor) and ids.shape == (1, 12)

    def test_combined_with_invoke_inside(self, gpt2):
        with torch.no_grad(), gpt2.trace() as tracer:
            with tracer.invoke(PROMPT):
                hidden = gpt2.transformer.h[0].output[0].save()
        assert hidden.shape[-1] == 768


class TestActivationModification:
    @torch.no_grad()
    def test_inplace_zeroing(self, gpt2):
        with gpt2.trace(PROMPT):
            pre = gpt2.transformer.h[-1].output[0].clone().save()
            gpt2.transformer.h[-1].output[0][:] = 0
            post = gpt2.transformer.h[-1].output[0].save()
        assert not (pre == 0).all()
        assert (post == 0).all()

    @torch.no_grad()
    def test_zeroing_layer_changes_prediction(self, gpt2):
        with gpt2.trace(PROMPT):
            ref = gpt2.output.logits[0, -1].argmax().save()
        with gpt2.trace(PROMPT):
            gpt2.transformer.h[-1].output[0][:] = 0
            modified = gpt2.output.logits[0, -1].argmax().save()
        assert ref.item() != modified.item()

    @torch.no_grad()
    def test_embedding_multiplication(self, gpt2):
        with gpt2.trace(PROMPT):
            pre = gpt2.transformer.wte.output.clone().save()
            gpt2.transformer.wte.output = gpt2.transformer.wte.output * 0
            post = gpt2.transformer.wte.output.save()
        assert not (pre == 0).all()
        assert (post == 0).all()


class TestGradients:
    def test_grad_capture_and_edit(self, gpt2):
        with gpt2.trace(PROMPT):
            a1 = gpt2.transformer.h[0].output
            loss = gpt2.output.logits.sum()
            with loss.backward():
                g1 = a1.grad.clone().save()
                a1.grad = a1.grad * 2  # replace the gradient flowing onward
                g2 = a1.grad.save()
        assert g1.shape[-1] == 768
        assert torch.equal(g1 * 2, g2)

    def test_grad_with_multiple_invokers(self, gpt2):
        """A gradient requested inside a batched invoke (two+ invokes).

        Each invoke's module output is a storage-sharing view of the full batch,
        which is not in the loss graph — a naive hook on it never fires. The grad
        must be provided and match the same input's single-invoke gradient.
        """
        # Reference: one invoke, no batching.
        with gpt2.trace() as tracer:
            with tracer.invoke(PROMPT):
                ref_x = gpt2.transformer.h[5].attn.c_proj.output
                with gpt2.lm_head.output.sum().backward():
                    ref_grad = ref_x.grad.save()

        # Two invokes -> invoke 2's activation is a batch-slice view.
        with gpt2.trace() as tracer:
            with tracer.invoke(PROMPT):
                gpt2.lm_head.output.save()
            with tracer.invoke(PROMPT):
                batched_x = gpt2.transformer.h[5].attn.c_proj.output
                with gpt2.lm_head.output.sum().backward():
                    batched_grad = batched_x.grad.save()

        assert batched_grad is not None
        assert batched_grad.shape == ref_grad.shape
        # Relative error (the summed-logit gradients are large, so an absolute
        # tolerance is meaningless; batched vs single differ only by fp noise).
        rel = (ref_grad - batched_grad).norm() / ref_grad.norm()
        assert rel < 1e-3, rel.item()

    def test_grad_edit_with_multiple_invokers(self, gpt2):
        """Editing a gradient inside a batched invoke is spliced back and re-read."""
        with gpt2.trace() as tracer:
            with tracer.invoke(PROMPT):
                gpt2.lm_head.output.save()
            with tracer.invoke(PROMPT):
                x = gpt2.transformer.h[5].attn.c_proj.output
                with gpt2.lm_head.output.sum().backward():
                    before = x.grad.clone().save()
                    x.grad = x.grad * 3.0
                    after = x.grad.save()
        assert torch.allclose(after, before * 3.0, atol=1e-4)


class TestAdhocModules:
    @torch.no_grad()
    def test_logit_lens(self, gpt2):
        # Apply lm_head to an intermediate hidden state, out of execution order.
        with gpt2.trace("The Eiffel Tower is in the city of"):
            hidden = gpt2.transformer.h[-1].output
            logits = gpt2.lm_head(gpt2.transformer.ln_f(hidden))
            tokens = torch.softmax(logits, dim=-1).argmax(dim=-1).save()
        assert (
            gpt2.tokenizer.decode(tokens[0])
            == "\n-el Tower is a the middle centre Paris"
        )


class TestInputSetting:
    @torch.no_grad()
    def test_zero_input(self, gpt2):
        with gpt2.trace(PROMPT):
            gpt2.transformer.h[1].input = gpt2.transformer.h[1].input * 0
            after = gpt2.transformer.h[1].input.save()
        assert (after == 0).all()


class TestSource:
    @torch.no_grad()
    def test_source_op_output(self, gpt2):
        with gpt2.trace(PROMPT):
            act = gpt2.transformer.h[0].mlp.source.self_act_0.output.save()
        assert act.ndim == 3  # (batch, seq, 4*hidden)


class TestEarlyStop:
    @torch.no_grad()
    def test_stop_halts_run(self, gpt2):
        with gpt2.trace(PROMPT) as tracer:
            first_layer = gpt2.transformer.h[0].output.save()
            tracer.stop()
        assert first_layer.shape[-1] == 768


class TestIteration:
    @torch.no_grad()
    def test_per_step_hidden_states(self, gpt2):
        with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            captured = nnsight.save([])
            for step in tracer.iter[:3]:
                captured.append(gpt2.transformer.h[0].output[0])
        assert len(captured) == 3
        # First step processes the full prompt; later (cached) steps process one token.
        assert captured[0].shape[0] > 1
        assert captured[1].shape[0] == 1

    @torch.no_grad()
    def test_with_form_matches_for_form(self, gpt2):
        # The deprecated `with tracer.iter[...]:` runs the block once per step, the
        # same as looping — the loop is just moved inside it.
        with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            via_for = nnsight.save([])
            for _ in tracer.iter[:3]:
                via_for.append(gpt2.transformer.h[0].output[0])
        with pytest.deprecated_call():
            with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
                via_with = nnsight.save([])
                with tracer.iter[:3]:
                    via_with.append(gpt2.transformer.h[0].output[0])
        assert len(via_with) == len(via_for) == 3
        assert all(torch.equal(a, b) for a, b in zip(via_with, via_for))

    @torch.no_grad()
    def test_with_form_open_ended_runs_to_the_last_step(self, gpt2):
        # `tracer.iter[:]` re-runs until the model stops, the same as the for form:
        # the over-run step is thrown into and caught, keeping the reached steps.
        with pytest.deprecated_call():
            with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
                captured = nnsight.save([])
                with tracer.iter[:]:
                    captured.append(gpt2.transformer.h[0].output[0])
        assert len(captured) == 3

    @torch.no_grad()
    def test_with_form_single_step(self, gpt2):
        with pytest.deprecated_call():
            with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
                box = nnsight.save({})
                with tracer.iter[1]:
                    box["x"] = gpt2.transformer.h[0].output[0]
        assert box["x"].shape[0] == 1  # a single cached decode step

    @torch.no_grad()
    def test_for_form_does_not_warn(self, gpt2):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with gpt2.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
                for _ in tracer.iter[:2]:
                    gpt2.transformer.h[0].output[0].save()


class TestSession:
    @torch.no_grad()
    def test_cross_trace_value_flow(self, gpt2):
        # Capture a hidden state in one trace, reuse it in the next — no explicit
        # .save() needed inside the session.
        with gpt2.session():
            with gpt2.trace(PROMPT):
                h0 = gpt2.transformer.h[0].output
            with gpt2.trace(PROMPT):
                diff = (gpt2.transformer.h[0].output - h0).abs().sum().save()
        assert diff.item() == 0.0  # same input -> identical


class TestCache:
    @torch.no_grad()
    def test_cache_all_outputs(self, gpt2):
        with gpt2.trace(PROMPT) as tracer:
            cache = tracer.cache()
        assert cache["model.transformer.h.0"].output is not None
        assert cache["model.transformer.h.0"].inputs is None

    @torch.no_grad()
    def test_cache_non_blocking_flag(self, gpt2):
        # non_blocking is a device-transfer speed flag; either setting yields the
        # same captured values (the default True is safe under single-stream runs).
        with gpt2.trace(PROMPT) as tracer:
            fast = tracer.cache(modules=[gpt2.transformer.h[0]], non_blocking=True)
        with gpt2.trace(PROMPT) as tracer:
            sync = tracer.cache(modules=[gpt2.transformer.h[0]], non_blocking=False)
        assert torch.equal(
            fast["model.transformer.h.0"].output,
            sync["model.transformer.h.0"].output,
        )

    @torch.no_grad()
    def test_cache_output_and_inputs(self, gpt2):
        with gpt2.trace(PROMPT) as tracer:
            cache = tracer.cache(include_inputs=True)
        assert torch.equal(
            cache["model.transformer.h.0"].output,
            cache["model.transformer.h.1"].inputs[0][0],
        )

    @torch.no_grad()
    def test_cache_generation_accumulates(self, gpt2):
        with gpt2.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            cache = tracer.cache(modules=[gpt2.transformer.h[0].attn.c_attn])
        assert len(cache["model.transformer.h.0.attn.c_attn"]) == 3

    @torch.no_grad()
    def test_cache_per_invoke_holds_its_own_rows(self, gpt2):
        # A cache opened inside an invoke records that invoke's rows, not the whole
        # batch — the same value the invoke's own read sees.
        with gpt2.trace() as tracer:
            with tracer.invoke("the cat sat"):
                ca = tracer.cache(modules=[gpt2.transformer.h[0].mlp])
                a = gpt2.transformer.h[0].mlp.output.save()
            with tracer.invoke("a much longer prompt appears now"):
                cb = tracer.cache(modules=[gpt2.transformer.h[0].mlp])
                b = gpt2.transformer.h[0].mlp.output.save()
        cached_a = ca["model.transformer.h.0.mlp"].output
        cached_b = cb["model.transformer.h.0.mlp"].output
        assert cached_a.shape[0] == 1 and cached_b.shape[0] == 1
        assert torch.equal(cached_a, a)
        assert torch.equal(cached_b, b)

    @torch.no_grad()
    def test_cache_captures_intervention(self, gpt2):
        with gpt2.trace(PROMPT) as tracer:
            cache = tracer.cache()
            gpt2.transformer.h[0].attn.c_attn.output = torch.zeros_like(
                gpt2.transformer.h[0].attn.c_attn.output
            )
        assert (cache["model.transformer.h.0.attn.c_attn"].output == 0).all()

    @torch.no_grad()
    def test_cache_alias_navigation(self):
        g = TransformersModel(
            "gpt2", task="text-generation", dispatch=True, rename={"mlp": "my_mlp"}
        )
        with g.trace(PROMPT) as tracer:
            cache = tracer.cache()
        assert torch.equal(
            cache.transformer.h[0].my_mlp.output,
            cache["model.transformer.h.0.mlp"].output,
        )

    @torch.no_grad()
    def test_cache_alias_full_navigation_and_index(self):
        # Full `cache.model....` paths, alias-renamed keys, and string indexing
        # into a renamed ModuleList entry.
        g = TransformersModel(
            "gpt2",
            task="text-generation",
            dispatch=True,
            rename={"transformer": "model", "h.0": "first_layer", "1": "second_layer"},
        )
        with g.trace(PROMPT) as tracer:
            cache = tracer.cache()
        assert torch.equal(
            cache["model.transformer.h.0"].output,
            cache.model.model.first_layer.output,
        )
        assert torch.equal(
            cache["model.transformer.h.1"].output,
            cache.model.model.h["second_layer"].output,
        )
        assert torch.equal(
            cache["model.model.first_layer"].output,
            cache.model.transformer.h[0].output,
        )


class TestRename:
    @torch.no_grad()
    def test_module_alias_everywhere(self):
        gpt2 = TransformersModel(
            "gpt2", task="text-generation", dispatch=True, rename={"mlp": "my_mlp"}
        )
        with gpt2.trace(PROMPT):
            via_alias = gpt2.transformer.h[0].my_mlp.output.save()
            via_name = gpt2.transformer.h[0].mlp.output.save()
        assert torch.equal(via_alias, via_name)

    @torch.no_grad()
    def test_modulelist_mount(self):
        gpt2 = TransformersModel(
            "gpt2", task="text-generation", dispatch=True, rename={"transformer.h": "layers"}
        )
        assert gpt2.layers[0] is gpt2.transformer.h[0]
        with gpt2.trace(PROMPT):
            out = gpt2.layers[0].output.save()
        assert out.shape[-1] == 768

    @torch.no_grad()
    def test_deep_path_mounts_on_root(self):
        gpt2 = TransformersModel(
            "gpt2", task="text-generation", dispatch=True,
            rename={"transformer.h.3.mlp": "my_mlp"},
        )
        assert gpt2.my_mlp is gpt2.transformer.h[3].mlp
        with gpt2.trace(PROMPT):
            mounted = gpt2.my_mlp.output.save()
            direct = gpt2.transformer.h[3].mlp.output.save()
        assert torch.equal(mounted, direct)

    @torch.no_grad()
    def test_leading_dot_module_list(self):
        # A leading dot is a no-op: `.h` aliases the `h` ModuleList as `layers`
        # wherever it resolves (under transformer).
        gpt2 = TransformersModel(
            "gpt2", task="text-generation", dispatch=True, rename={".h": "layers"}
        )
        assert gpt2.transformer.layers[0] is gpt2.transformer.h[0]
        with gpt2.trace(PROMPT):
            out = gpt2.transformer.layers[1].mlp.output.save()
        assert out.shape[-1] == 768

    @torch.no_grad()
    def test_forward_call_via_alias(self):
        # Calling an aliased module runs the same forward as calling it by name.
        gpt2 = TransformersModel(
            "gpt2", task="text-generation", dispatch=True,
            rename={"transformer.h.0.mlp": "my_mlp"},
        )
        hidden = torch.randn(1, 4, 768)
        assert torch.equal(gpt2.my_mlp(hidden), gpt2.transformer.h[0].mlp(hidden))

    @torch.no_grad()
    def test_input_via_alias(self):
        # Reading `.input` through an alias matches reading it through the name.
        gpt2 = TransformersModel(
            "gpt2", task="text-generation", dispatch=True, rename={"mlp": "ffn"}
        )
        with gpt2.trace(PROMPT):
            via_alias = gpt2.transformer.h[0].ffn.input.save()
            via_name = gpt2.transformer.h[0].mlp.input.save()
        assert torch.equal(via_alias, via_name)


@pytest.fixture(scope="module")
def gpt2_meta():
    # Undispatched: architecture on meta, no real weights. scan never dispatches
    # it, so it's safe to share this instance across the scan tests.
    return TransformersModel("gpt2", task="text-generation")


class TestScan:
    @torch.no_grad()
    def test_scan_reads_shapes_without_weights(self, gpt2_meta):
        assert gpt2_meta.dispatched is False
        with gpt2_meta.scan(PROMPT):
            hs = gpt2_meta.transformer.h[-1].output[0].save()
        assert hs.shape[-1] == 768
        # scanning only propagates shapes: the model is never loaded.
        assert gpt2_meta.dispatched is False
        assert all(p.device.type == "meta" for p in gpt2_meta._module.parameters())

    @torch.no_grad()
    def test_scan_values_are_fake(self, gpt2_meta):
        with gpt2_meta.scan(PROMPT):
            out = gpt2_meta.transformer.wte.output.save()
        assert "Fake" in type(out).__name__

    @torch.no_grad()
    def test_scan_shapes_match_real_trace(self, gpt2):
        # Fidelity: shapes seen in a fake scan match a real forward pass.
        with gpt2.scan(PROMPT):
            scanned = gpt2.transformer.h[-1].output[0].save()
        with gpt2.trace(PROMPT):
            traced = gpt2.transformer.h[-1].output[0].save()
        assert tuple(scanned.shape) == tuple(traced.shape)

    @torch.no_grad()
    def test_scan_does_not_dispatch_but_trace_does(self):
        model = TransformersModel("gpt2", task="text-generation")
        with model.scan(PROMPT):
            model.transformer.h[0].output[0].save()
        assert model.dispatched is False
        with model.trace(PROMPT):
            real = model.transformer.h[0].output[0].save()
        assert model.dispatched is True
        assert real.device.type != "meta"
        assert torch.isfinite(real).all()

    @pytest.mark.parametrize("key", ["dtype", "torch_dtype"])
    @torch.no_grad()
    def test_dtype_auto_builds_on_meta(self, key):
        """`dtype="auto"` must not break the lazy build.

        It means "read the dtype off the checkpoint weights", which only
        from_pretrained can do — the meta build goes through from_config, where a
        string dtype is resolved with `getattr(torch, dtype)` and "auto" raises.
        It is a natural value to carry from a config file, and it works on the
        dispatch path, so it would fail only when `dispatch` is flipped off.
        """
        model = TransformersModel("gpt2", task="text-generation", **{key: "auto"})
        assert model.dispatched is False
        # The architecture is there, on meta, with no weights loaded.
        assert len(model.transformer.h) == model.config.n_layer
        assert all(p.device.type == "meta" for p in model._module.parameters())
        # "auto" must not survive onto the config either: from_config would take
        # it as its default and hit the same getattr(torch, "auto").
        assert model.config.dtype != "auto"

    @torch.no_grad()
    def test_explicit_dtype_still_reaches_the_meta_build(self):
        """Dropping "auto" must not drop a real dtype."""
        model = TransformersModel(
            "gpt2", task="text-generation", dtype=torch.float16
        )
        assert all(p.dtype == torch.float16 for p in model._module.parameters())


def _has_lora(model) -> bool:
    return any("lora" in name.lower() for name, _ in model._module.named_modules())


peft_installed = importlib.util.find_spec("peft") is not None


@pytest.fixture(scope="module")
def lora_adapter(tmp_path_factory):
    # Build a tiny LoRA adapter over gpt2 and save it locally, so PEFT tests run
    # offline against a real adapter (adapter_config.json + weights on disk).
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    path = tmp_path_factory.mktemp("lora_adapter")
    base = AutoModelForCausalLM.from_pretrained("gpt2")
    config = LoraConfig(task_type="CAUSAL_LM", target_modules=["c_attn"], r=4)
    get_peft_model(base, config).save_pretrained(str(path))
    return str(path)


@pytest.mark.skipif(not peft_installed, reason="peft is not installed")
class TestPeft:
    def test_meta_load_grafts_adapter(self, lora_adapter):
        # Undispatched: the adapter's architecture is grafted onto the meta model
        # (so remote module paths match) without loading real weights.
        model = TransformersModel("gpt2", task="text-generation", peft=lora_adapter)
        assert model.dispatched is False
        assert _has_lora(model)
        assert all(p.device.type == "meta" for p in model._module.parameters())

    def test_get_env_reports_peft(self, lora_adapter):
        model = TransformersModel("gpt2", task="text-generation", peft=lora_adapter)
        assert model._remoteable_get_env() == {"peft": lora_adapter}

    def test_get_env_empty_without_peft(self, gpt2):
        assert gpt2._remoteable_get_env() == {}

    @torch.no_grad()
    def test_dispatch_and_trace_with_peft(self, lora_adapter):
        model = TransformersModel(
            "gpt2", task="text-generation", peft=lora_adapter, dispatch=True
        )
        assert _has_lora(model)
        with model.trace(PROMPT):
            out = model.transformer.h[0].attn.output[0].save()
        assert out.shape[-1] == 768
        assert torch.isfinite(out).all()

    @torch.no_grad()
    def test_set_env_transitions(self, lora_adapter):
        # Server-side per-request swap: load, no-op, unload — each leaves a
        # traceable model with the right adapter state.
        model = TransformersModel("gpt2", task="text-generation", dispatch=True)
        assert model.peft is None and not _has_lora(model)

        model._remoteable_set_env({"peft": lora_adapter})  # None -> X
        assert model.peft == lora_adapter and _has_lora(model)

        module_after_load = model._module
        model._remoteable_set_env({"peft": lora_adapter})  # X -> X (no-op)
        assert model._module is module_after_load

        model._remoteable_set_env({})  # X -> None
        assert model.peft is None and not _has_lora(model)

        with model.trace(PROMPT):
            out = model.transformer.h[0].attn.output[0].save()
        assert out.shape[-1] == 768


def _hidden(block_output):
    """A gpt2 block's output is a tuple (string path) or a bare tensor (raw
    input_ids path); return the hidden-state tensor either way."""
    return block_output[0] if isinstance(block_output, tuple) else block_output


class TestTokenization:
    """Input forms TransformersModel accepts. Every invoke — strings, token-id
    lists, tensors, or a pre-tokenized encoding — is normalized to per-row
    ``input_ids`` and left-pad batched into one forward, so mixed formats and
    unequal lengths combine freely.
    """

    @torch.no_grad()
    def test_list_of_strings_is_a_batch(self, gpt2):
        with gpt2.trace(["the cat sat here", "a dog ran now"]):
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[0] == 2

    @torch.no_grad()
    def test_unequal_length_list_is_padded(self, gpt2):
        # Different token lengths are left-padded to a common length, not rejected.
        with gpt2.trace(["hi", "a much longer prompt appears here"]):
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[0] == 2

    @torch.no_grad()
    def test_token_id_list(self, gpt2):
        ids = gpt2.tokenizer("the quick brown fox").input_ids  # list[int]
        with gpt2.trace(ids):
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[1] == len(ids)

    @torch.no_grad()
    def test_list_of_token_id_lists(self, gpt2):
        a = gpt2.tokenizer("the quick brown fox").input_ids
        b = gpt2.tokenizer("a dog ran").input_ids
        with gpt2.trace([a, b]):  # list[list[int]]
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[0] == 2

    @torch.no_grad()
    def test_one_dimensional_tensor(self, gpt2):
        ids = gpt2.tokenizer("the quick brown fox", return_tensors="pt").input_ids[0]
        assert ids.ndim == 1
        with gpt2.trace(ids):
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[0] == 1  # one row, not one row per token
        assert _hidden(out).shape[1] == ids.shape[0]

    @torch.no_grad()
    def test_batch_encoding_positional(self, gpt2):
        enc = gpt2.tokenizer("the quick brown fox", return_tensors="pt")
        with gpt2.trace(enc):  # passed positionally, not unpacked
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[1] == enc.input_ids.shape[1]

    @torch.no_grad()
    def test_input_ids_keyword(self, gpt2):
        ids = gpt2.tokenizer("the quick brown fox", return_tensors="pt").input_ids
        with gpt2.trace(input_ids=ids):
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[1] == ids.shape[1]

    @torch.no_grad()
    def test_input_ids_and_attention_mask_keyword(self, gpt2):
        enc = gpt2.tokenizer("the quick brown fox", return_tensors="pt")
        with gpt2.trace(input_ids=enc.input_ids, attention_mask=enc.attention_mask):
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[1] == enc.input_ids.shape[1]

    @torch.no_grad()
    def test_raw_tensor_input(self, gpt2):
        ids = gpt2.tokenizer("the quick brown fox", return_tensors="pt").input_ids
        with gpt2.trace(ids):
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[1] == ids.shape[1]

    @torch.no_grad()
    def test_batch_encoding_unpacked(self, gpt2):
        enc = gpt2.tokenizer("the quick brown fox", return_tensors="pt")
        with gpt2.trace(**enc):
            out = gpt2.transformer.h[0].output.save()
        assert _hidden(out).shape[1] == enc.input_ids.shape[1]


class TestSkipRealModel:
    @torch.no_grad()
    def test_skip_mlp_changes_prediction(self, gpt2):
        # Capture the mlp output shape first (reading it runs the module, so the
        # skip must happen in a separate trace).
        with gpt2.trace(PROMPT):
            mlp_out = gpt2.transformer.h[0].mlp.output.save()
        with gpt2.trace(PROMPT):
            base = gpt2.output.logits.save()
        with gpt2.trace(PROMPT):
            gpt2.transformer.h[0].mlp.skip(torch.zeros_like(mlp_out))
            skipped = gpt2.output.logits.save()
        assert not torch.allclose(base, skipped)

    @torch.no_grad()
    def test_skipped_module_output_is_replacement(self, gpt2):
        with gpt2.trace(PROMPT):
            mlp_out = gpt2.transformer.h[0].mlp.output.save()
        with gpt2.trace(PROMPT):
            gpt2.transformer.h[0].mlp.skip(torch.zeros_like(mlp_out))
            out = gpt2.transformer.h[0].mlp.output.save()
        assert torch.all(out == 0)


class TestMultipleBackward:
    def test_two_backward_passes_differ(self, gpt2):
        with gpt2.trace(PROMPT):
            a1 = gpt2.transformer.h[0].output
            logits = gpt2.output.logits
            with logits.sum().backward(retain_graph=True):
                g1 = a1.grad.clone().save()
            with (logits.sum() * 3).backward():
                g2 = a1.grad.clone().save()
        # Different objective on each pass -> different gradients (grads are not
        # accumulated across the two backward contexts).
        assert not torch.allclose(g1, g2)


class TestModernDecoder:
    """A non-gpt2 decoder (tiny Llama, RoPE/GQA/RMSNorm): trace, generate, source,
    and batching all work across a different tree and layer-output structure."""

    @torch.no_grad()
    def test_tree_and_bare_tensor_output(self, llama):
        assert hasattr(llama, "model") and hasattr(llama, "lm_head")
        with llama.trace("hello world"):
            out = llama.model.layers[0].output.save()
        # A Llama decoder layer returns its hidden states directly, not a tuple.
        assert isinstance(out, torch.Tensor) and out.ndim == 3

    @torch.no_grad()
    def test_edit_layer_changes_logits(self, llama):
        with llama.trace("hello world"):
            base = llama.output.logits.save()
        with llama.trace("hello world"):
            llama.model.layers[0].output[:] = 0
            edited = llama.output.logits.save()
        assert not torch.allclose(base, edited)

    @torch.no_grad()
    def test_generate_returns_ids(self, llama):
        with llama.generate("hello", max_new_tokens=3) as tracer:
            ids = tracer.result.save()
        assert ids.ndim == 2 and ids.shape[0] == 1

    def test_source_instrumentation(self, llama):
        # The forward's operations are exposed for source tracing (a different AST
        # from gpt2's forward).
        ops = str(llama.model.layers[0].source)
        assert "self_" in ops or "torch_" in ops

    @torch.no_grad()
    def test_batched_narrowing(self, llama):
        with llama.trace() as tracer:
            with tracer.invoke("hi there"):
                a = llama.model.layers[0].output.save()
            with tracer.invoke("a b c d e"):
                b = llama.model.layers[0].output.save()
        # Each invoke is narrowed to its own row on the batch dim.
        assert a.shape[0] == 1 and b.shape[0] == 1


class TestCustomEnvoys:
    """`envoys=` maps a module type or path suffix to a custom Envoy subclass, so a
    custom eproperty (a per-head view here) attaches to specific modules."""

    @pytest.fixture(scope="class")
    def heads_model(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

        return TransformersModel(
            "gpt2", task="text-generation", dispatch=True, envoys={GPT2MLP: Heads}
        )

    def test_type_key_maps_module_to_subclass(self, heads_model):
        assert isinstance(heads_model.transformer.h[0].mlp, Heads)
        # A module not named by the map stays the base Envoy.
        assert type(heads_model.transformer.h[0].attn) is Envoy

    def test_string_key_maps_by_path_suffix(self):
        model = TransformersModel(
            "gpt2", task="text-generation", dispatch=True, envoys={"mlp": Heads}
        )
        assert isinstance(model.transformer.h[2].mlp, Heads)

    @torch.no_grad()
    def test_custom_eproperty_reads_per_head_view(self, heads_model):
        with heads_model.trace("hello world"):
            view = heads_model.transformer.h[0].mlp.heads.save()
        assert view.ndim == 4 and view.shape[1] == Heads.n_heads

    @torch.no_grad()
    def test_transform_writes_edited_view_back(self, heads_model):
        with heads_model.trace("hello world"):
            base = heads_model.output.logits.save()
        with heads_model.trace("hello world"):
            heads_model.transformer.h[0].mlp.heads[:, 5] = 0  # zero head 5
            edited = heads_model.output.logits.save()
        assert not torch.allclose(base, edited)
