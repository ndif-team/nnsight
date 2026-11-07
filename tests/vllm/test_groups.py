"""A group: one block over several prompts, ``tracer.invoke([p1, ..., pn])``.

The members are separate vLLM requests sharing one block, which at every step
is served the rows of the members in that step concatenated in member order,
learns their layout from ``tracer.group``, and has its writes scattered back to
each member's rows. These tests pin that a group reads what the same prompts
read in their own invokes, writes only where it says, follows its members
through a generation as they drop out, comes home once, and refuses what it
cannot carry.

The engine runs with ``VLLM_BATCH_INVARIANT=1`` so a row computed beside other
requests is bit-for-bit the row computed alone; without it the kernels' reduction
order follows the batch.
"""

import os

import pytest
import torch

pytest.importorskip("vllm")

import nnsight

ET = "The Eiffel Tower is located in the city of"
HELLO = "Hello"
MSG = "Madison Square Garden is located in"
JAPAN = "The capital of Japan is"
PROMPTS = [ET, HELLO, MSG]


@pytest.fixture(scope="module")
def model():
    from nnsight.modeling.vllm import VLLM

    if torch.cuda.device_count() < 1:
        pytest.skip("vLLM tests need a GPU")
    # The engine core inherits the environment when it is spawned, so the flag
    # only has to be set while this engine is built.
    previous = os.environ.get("VLLM_BATCH_INVARIANT")
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    try:
        return VLLM("gpt2", gpu_memory_utilization=0.1, dtype="float32", dispatch=True)
    finally:
        if previous is None:
            del os.environ["VLLM_BATCH_INVARIANT"]
        else:
            os.environ["VLLM_BATCH_INVARIANT"] = previous


def _lengths(model, prompts):
    return [len(model.tokenizer.encode(prompt)) for prompt in prompts]


def _solo(model, prompt):
    """The last-token row at layer 8 and the logits of ``prompt`` in its own trace."""
    with model.trace(prompt, temperature=0.0, max_tokens=1):
        row = model.transformer.h[8].output[-1].clone().save()
        logits = model.logits.save()
    return row, logits[0]


def _count(model):
    return model.vllm_entrypoint.collective_rpc("nnsight_request_count")[0]


class TestServing:
    @torch.no_grad()
    def test_members_rows_are_served_concatenated_in_member_order(self, model):
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPTS):
                members = nnsight.save(tracer.group)
                hidden = model.transformer.h[8].output.save()
                logits = model.logits.save()

        lengths = _lengths(model, PROMPTS)
        starts = [sum(lengths[:index]) for index in range(len(lengths))]
        assert [(m.index, m.start, m.size, m.step) for m in members] == [
            (index, start, size, 0) for index, (start, size) in enumerate(zip(starts, lengths))
        ]
        assert hidden.shape == (sum(lengths), 768)
        assert logits.shape == (3, model.tokenizer.vocab_size)
        for member, prompt in zip(members, PROMPTS):
            row, solo_logits = _solo(model, prompt)
            assert torch.equal(hidden[member.start + member.size - 1], row)
            assert torch.equal(logits[member.index], solo_logits)

    @torch.no_grad()
    def test_a_write_lands_on_its_member_only(self, model):
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            for prompt in PROMPTS:
                with tracer.invoke(prompt):
                    clean = model.logits[0].save()

        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPTS):
                member = tracer.group[1]
                model.transformer.h[8].output[member.start + member.size - 1] = 0
                edited = model.logits.save()

        assert torch.equal(edited[0], clean[0])
        assert not torch.equal(edited[1], clean[1])
        assert torch.equal(edited[2], clean[2])

    @torch.no_grad()
    def test_an_in_place_edit_and_a_swap_both_land(self, model):
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPTS):
                model.transformer.h[8].output[:] = 0
                after_edit = model.transformer.h[9].input.save()
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPTS):
                model.transformer.h[8].output = torch.zeros_like(model.transformer.h[8].output)
                after_swap = model.transformer.h[9].input.save()
        assert torch.all(after_edit == 0)
        assert torch.all(after_swap == 0)

    @torch.no_grad()
    def test_the_layout_read_after_a_value_is_the_same_steps(self, model):
        with model.trace(temperature=0.0, max_tokens=3) as tracer:
            with tracer.invoke(PROMPTS):
                pairs = list().save()
                for step in tracer.iter[:3]:
                    rows = model.transformer.h[8].output.shape[0]
                    pairs.append((rows, sum(m.size for m in tracer.group)))
        assert [rows for rows, _ in pairs] == [sum(_lengths(model, PROMPTS)), 3, 3]
        assert all(rows == layout for rows, layout in pairs)

    @torch.no_grad()
    def test_a_trace_given_a_list_is_a_group(self, model):
        with model.trace(PROMPTS, temperature=0.0, max_tokens=1) as tracer:
            members = nnsight.save(tracer.group)
            logits = model.logits.save()
        assert [m.index for m in members] == [0, 1, 2]
        assert logits.shape[0] == 3

    @torch.no_grad()
    def test_a_tokenizer_batch_is_a_group(self, model):
        batch = model.tokenizer(PROMPTS, padding=True, return_tensors="pt")
        with model.trace(batch, temperature=0.0, max_tokens=1) as tracer:
            members = nnsight.save(tracer.group)
        assert [m.size for m in members] == _lengths(model, PROMPTS)

    @torch.no_grad()
    def test_a_group_and_invokes_share_a_trace(self, model):
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke([ET, MSG]):
                logits = model.logits.save()
            with tracer.invoke(HELLO):
                logits = model.logits.save()
        group, single = logits
        assert group.shape[0] == 2 and single.shape[0] == 1
        assert torch.equal(single[0], _solo(model, HELLO)[1])


class TestMixedSteps:
    @torch.no_grad()
    def test_a_group_too_big_for_a_step_prefills_across_steps(self, model):
        budget = model.vllm_entrypoint.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens
        prompt = [464] * 1000  # 1000 tokens of " The"
        members = budget // 1000 + 1
        with model.trace(temperature=0.0, max_tokens=3, ignore_eos=True) as tracer:
            with tracer.invoke([[*prompt[:-1], 100 + index] for index in range(members)]):
                layouts = list().save()
                sizes = list().save()
                for step in tracer.iter[:]:
                    layouts.append(tracer.group)
                    sizes.append(model.transformer.h[8].output.shape[0])

        assert sorted(m.index for m in layouts[0]) == list(range(len(layouts[0])))
        assert len(layouts[0]) < members
        mixed = layouts[1]
        assert {m.step for m in mixed} == {0, 1}
        for layout, size in zip(layouts, sizes):
            assert [m.start for m in layout] == [sum(m.size for m in layout[:i]) for i in range(len(layout))]
            assert sum(m.size for m in layout) == size
            assert all(m.size == (1000 if m.step == 0 else 1) for m in layout)
        prefills = [m.index for layout in layouts for m in layout if m.step == 0]
        assert sorted(prefills) == list(range(members))


class TestGeneration:
    @torch.no_grad()
    def test_the_layout_follows_the_members_still_running(self, model):
        # ET's greedy continuation is " Paris, France." — "." (token 13) ends it
        # at its fourth token; the others run to max_tokens.
        prompts = [ET, MSG, JAPAN]
        with model.trace(temperature=0.0, max_tokens=8, ignore_eos=True, stop_token_ids=[13]) as tracer:
            with tracer.invoke(prompts):
                layouts = list().save()
                samples = list().save()
                for step in tracer.iter[:8]:
                    layouts.append(tracer.group)
                    samples.append(model.samples)
                result = tracer.result.save()

        assert isinstance(layouts, list) and len(layouts) == 8
        assert [(m.index, m.step) for m in layouts[0]] == [(0, 0), (1, 0), (2, 0)]
        stopped = len(result[0].outputs[0].token_ids)
        assert stopped == 4
        present = [step for step, layout in enumerate(layouts) if 0 in [m.index for m in layout]]
        # A step past its stop is allowed: vLLM schedules a step before the last
        # one's token is known (async scheduling).
        assert present == list(range(len(present))) and stopped <= len(present) <= stopped + 1
        for step, (layout, drawn) in enumerate(zip(layouts, samples)):
            assert [m.index for m in layout if m.index] == [1, 2]
            assert drawn.shape == (len(layout), 1)
            for row, member in enumerate(layout):
                assert member.step == step
                tokens = result[member.index].outputs[0].token_ids
                if member.step < len(tokens):
                    assert drawn[row, 0].item() == tokens[member.step]

    @torch.no_grad()
    def test_result_is_every_members_output_in_member_order(self, model):
        with model.trace(temperature=0.0, max_tokens=3) as tracer:
            with tracer.invoke(PROMPTS):
                result = tracer.result.save()
        assert isinstance(result, list) and len(result) == 3
        assert [output.prompt for output in result] == PROMPTS
        assert result[0].outputs[0].text == " Paris, France"

    @torch.no_grad()
    def test_a_saved_name_comes_home_once(self, model):
        with model.trace(temperature=0.0, max_tokens=4, ignore_eos=True) as tracer:
            with tracer.invoke(PROMPTS):
                total = nnsight.save([0])
                for step in tracer.iter[:]:
                    total[0] += len(tracer.group)
        assert total == [12]


class TestErrors:
    @torch.no_grad()
    def test_a_block_error_ends_every_member(self, model):
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            with model.trace(temperature=0.0, max_tokens=12) as tracer:
                with tracer.invoke([MSG, JAPAN]):
                    for step in tracer.iter[:]:
                        model.samples
                        if step == 1:
                            1 / 0
        assert _count(model) == 0
        with model.trace(temperature=0.0, max_tokens=3) as tracer:
            with tracer.invoke([MSG, JAPAN]):
                result = tracer.result.save()
        assert [output.outputs[0].text for output in result] == [" the heart of", " the capital of"]

    @torch.no_grad()
    def test_a_stop_ends_every_member(self, model):
        # The engine stops scheduling the members, so far fewer than max_tokens
        # steps run — counted by a cache that watches every step until they end.
        with model.trace(temperature=0.0, max_tokens=20) as tracer:
            with tracer.invoke([MSG, JAPAN]):
                cache = tracer.cache(modules=[model.transformer.h[0]])
                for step in tracer.iter[:20]:
                    model.logits
                    if step == 1:
                        tracer.stop()
            with tracer.invoke(HELLO, max_tokens=12, ignore_eos=True):
                result = tracer.result.save()
        steps = cache["model.transformer.h.0"]
        assert len(steps) < 20
        assert steps.output[0].shape[0] == sum(_lengths(model, [MSG, JAPAN]))
        assert len(result.outputs[0].token_ids) == 12
        assert _count(model) == 0

    @torch.no_grad()
    def test_a_group_skip_is_refused_and_the_engine_survives(self, model):
        with pytest.raises(RuntimeError, match="not supported in a group"):
            with model.trace(temperature=0.0, max_tokens=1) as tracer:
                with tracer.invoke(PROMPTS):
                    model.transformer.h[3].skip(model.transformer.h[3].input)
        with model.trace(ET, temperature=0.0, max_tokens=1):
            logits = model.logits.save()
        assert model.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_tracer_group_outside_a_group_is_refused(self, model):
        with pytest.raises(RuntimeError, match="tracer.group is a group's layout"):
            with model.trace(ET, temperature=0.0, max_tokens=1) as tracer:
                members = tracer.group

    def test_n_is_refused_on_a_group(self, model):
        with pytest.raises(ValueError, match="n=2 is not supported on a group"):
            with model.trace(temperature=1.0, max_tokens=1, n=2) as tracer:
                with tracer.invoke(PROMPTS):
                    model.logits.save()

    def test_a_per_member_setting_is_refused(self, model):
        with pytest.raises(ValueError, match="max_tokens=\\[1, 2, 3\\] cannot be split"):
            with model.trace(temperature=0.0) as tracer:
                with tracer.invoke(PROMPTS, max_tokens=[1, 2, 3]):
                    model.logits.save()
        with pytest.raises(ValueError, match="seed=\\[0, 1, 2\\] cannot be split"):
            with model.trace(temperature=1.0, seed=[0, 1, 2]) as tracer:
                with tracer.invoke(PROMPTS):
                    model.logits.save()

    def test_lora_is_refused_on_a_group(self, model):
        with pytest.raises(ValueError, match="lora_request is not supported on a group"):
            with model.trace(temperature=0.0) as tracer:
                with tracer.invoke(PROMPTS, lora_request=object()):
                    model.logits.save()

    def test_a_list_inside_the_list_is_refused(self, model):
        with pytest.raises(ValueError, match="one flat list of prompts"):
            with model.trace(temperature=0.0) as tracer:
                with tracer.invoke([ET, [HELLO, MSG]]):
                    model.logits.save()

    @torch.no_grad()
    def test_nothing_is_left_behind(self, model):
        assert _count(model) == 0
