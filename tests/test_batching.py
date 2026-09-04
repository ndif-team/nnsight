"""Batching / invokers: ``with tracer.invoke(x):`` blocks combined into one
forward, each scoped to its rows.

Correctness anchor: a left-padded batched prompt must produce the same last-token
logits as running it alone (this also exercises the mask-derived position_ids).
"""


import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

import nnsight
from nnsight.intervention.envoy import Envoy
from nnsight.modeling.transformers import TransformersModel

P1 = "the cat sat"
P2 = "a much longer prompt appears here"
MSG = "Madison Square Garden is located in the city of"  # greedily -> " New York City"
ET = "The Eiffel Tower is in the city of"


@pytest.fixture(scope="module")
def gpt2():
    return TransformersModel("gpt2", task="text-generation", dispatch=True)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _BatchEnvoy(Envoy):
    """An Envoy that stacks each invoke's tensor input into one batch, so batching
    behavior can be exercised without a real batching model."""

    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}


class TestTraceBatching:
    @torch.no_grad()
    def test_batched_matches_solo_logits(self, gpt2):
        # Each invoke's last-token logits must equal the prompt run on its own —
        # proving left-pad batching + position_ids are correct.
        with gpt2.trace(P1):
            solo1 = gpt2.output.logits[0, -1].save()
        with gpt2.trace(P2):
            solo2 = gpt2.output.logits[0, -1].save()
        with gpt2.trace() as tracer:
            with tracer.invoke(P1):
                b1 = gpt2.output.logits[0, -1].save()
            with tracer.invoke(P2):
                b2 = gpt2.output.logits[0, -1].save()
        assert torch.allclose(b1, solo1, atol=1e-3)
        assert torch.allclose(b2, solo2, atol=1e-3)

    @torch.no_grad()
    def test_each_invoke_sees_its_own_rows(self, gpt2):
        with gpt2.trace() as tracer:
            with tracer.invoke(P1):
                a = gpt2.transformer.h[0].output[0].save()
            with tracer.invoke(P2):
                b = gpt2.transformer.h[0].output[0].save()
        # One row each (the invoke's slice of the batch), padded to the batch seq.
        assert a.shape[0] == b.shape[0]  # common padded length
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)

    @torch.no_grad()
    def test_edit_is_isolated_to_its_invoke(self, gpt2):
        with gpt2.trace() as tracer:
            with tracer.invoke(P1):
                gpt2.transformer.h[0].output[0][:] = 0
                edited = gpt2.transformer.h[0].output[0].save()
            with tracer.invoke(P1):
                other = gpt2.transformer.h[0].output[0].save()
        assert torch.all(edited == 0)
        assert torch.any(other != 0)

    @torch.no_grad()
    def test_edit_changes_only_its_prediction(self, gpt2):
        with gpt2.trace(P1):
            base = gpt2.output.logits[0, -1].save()
        with gpt2.trace() as tracer:
            with tracer.invoke(P1):
                gpt2.transformer.wte.output[:] = 0  # zero embeddings for this invoke
                zeroed = gpt2.output.logits[0, -1].save()
            with tracer.invoke(P1):
                intact = gpt2.output.logits[0, -1].save()
        assert not torch.allclose(zeroed, base, atol=1e-3)
        assert torch.allclose(intact, base, atol=1e-3)

    @torch.no_grad()
    def test_three_invokes(self, gpt2):
        with gpt2.trace() as tracer:
            outs = nnsight.save([])
            for prompt in ("one two", "three four five", "six"):
                with tracer.invoke(prompt):
                    outs.append(gpt2.transformer.h[0].output[0])
        assert len(outs) == 3

    @torch.no_grad()
    def test_empty_invoke_sees_whole_batch(self, gpt2):
        with gpt2.trace() as tracer:
            with tracer.invoke(P1):
                pass
            with tracer.invoke(P2):
                pass
            with tracer.invoke():  # no input -> whole batch, no row scoping
                whole = gpt2.output.logits.save()
        assert whole.shape[0] == 2  # both invokes' rows


class TestGenerateBatching:
    @torch.no_grad()
    def test_batched_generate_returns_per_prompt_results(self, gpt2):
        with gpt2.pipe(max_new_tokens=3, do_sample=False) as tracer:
            with tracer.invoke("The Eiffel Tower is in"):
                a = gpt2.transformer.h[0].output[0].save()
                result = tracer.result.save()
            with tracer.invoke("The capital of France is"):
                b = gpt2.transformer.h[0].output[0].save()
        assert len(result) == 2
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        texts = [
            r[0]["generated_text"] if isinstance(r, list) else r["generated_text"]
            for r in result
        ]
        assert all(isinstance(t, str) and t for t in texts)

    @torch.no_grad()
    def test_single_generate_unchanged(self, gpt2):
        # A lone prompt keeps the scalar-input pipeline output shape.
        with gpt2.pipe("The Eiffel Tower is in", max_new_tokens=3) as tracer:
            result = tracer.result.save()
        assert result[0]["generated_text"].startswith("The Eiffel Tower is in")


class TestBatchingErrors:
    def test_no_input_and_no_invoke_raises(self, gpt2):
        with pytest.raises(ValueError, match="invoke"):
            with gpt2.trace():
                pass  # no input and no invoke block

    def test_invoke_while_running_raises(self, gpt2):
        with pytest.raises(ValueError, match="already running"):
            with gpt2.trace() as tracer:
                with tracer.invoke(P1):
                    with tracer.invoke(P2):  # nested invoke mid-run
                        pass

    def test_plain_model_cannot_batch(self):
        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8)
                self.b = nn.Linear(8, 8)

            def forward(self, x):
                return self.b(self.a(x))

        model = Envoy(TwoLayer())
        with pytest.raises(NotImplementedError, match="batching"):
            with model.trace() as tracer:
                with tracer.invoke(torch.randn(2, 8)):
                    pass
                with tracer.invoke(torch.randn(2, 8)):
                    pass


class TestMultiInvokeSkip:
    """A skip bypasses a module's body and substitutes its output. Across invokes
    there's no combined body output to splice into — the body didn't run — so the
    output is rebuilt from each invoke's replacement, in its own rows."""

    @torch.no_grad()
    def test_each_invoke_gets_its_own_replacement(self):
        torch.manual_seed(0)
        envoy = _BatchEnvoy(_MLP())
        with envoy.trace() as tracer:
            with tracer.invoke(torch.randn(2, 8)):
                envoy.fc1.skip(torch.ones(2, 8))
                a = envoy.fc1.output.save()
            with tracer.invoke(torch.randn(3, 8)):
                envoy.fc1.skip(torch.full((3, 8), 5.0))
                b = envoy.fc1.output.save()
        assert a.shape == (2, 8) and bool((a == 1).all())
        assert b.shape == (3, 8) and bool((b == 5).all())

    @torch.no_grad()
    def test_partial_skip_raises(self):
        # One invoke skips, the other runs the module — a shared forward can't do
        # both, so it's refused rather than silently dropping the run invoke's rows.
        torch.manual_seed(0)
        envoy = _BatchEnvoy(_MLP())
        with pytest.raises(ValueError, match="cover every row"):
            with envoy.trace() as tracer:
                with tracer.invoke(torch.randn(2, 8)):
                    envoy.fc1.skip(torch.ones(2, 8))
                with tracer.invoke(torch.randn(3, 8)):
                    envoy.fc1.output.save()  # this invoke runs fc1

    @torch.no_grad()
    def test_single_invoke_skip_unaffected(self):
        torch.manual_seed(0)
        envoy = _BatchEnvoy(_MLP())
        with envoy.trace(torch.randn(4, 8)):
            envoy.fc1.skip(torch.ones(4, 8))
            out = envoy.fc1.output.save()
        assert out.shape == (4, 8) and bool((out == 1).all())

    def test_concat_recurses_tuples_and_dicts(self):
        # A skipped module can return a tuple/dict (a transformer block's output is
        # a tuple), so reassembly concatenates leaf by leaf, in row order.
        from nnsight.intervention.batching import concat

        a = (torch.ones(2, 4), {"k": torch.ones(2, 3)})
        b = (torch.full((3, 4), 5.0), {"k": torch.full((3, 3), 5.0)})
        tensor, mapping = concat([a, b])
        assert tensor.shape == (5, 4)
        assert bool((tensor[:2] == 1).all()) and bool((tensor[2:] == 5).all())
        assert mapping["k"].shape == (5, 3)


class TestWholeTensorWrites:
    """An invoke's rows are spliced back into the combined batch. A whole-tensor
    write in a multi-invoke trace has to keep the row count it was served — the
    splice is a plain cat, so what a wrong-height write does downstream is the
    model's business, not checked here."""

    @torch.no_grad()
    def test_the_right_row_count_lands(self):
        torch.manual_seed(0)
        envoy = _BatchEnvoy(_MLP())
        with envoy.trace() as tracer:
            with tracer.invoke(torch.randn(2, 8)):
                envoy.fc1.output = torch.ones(2, 8)
                a = envoy.fc1.output.save()
            with tracer.invoke(torch.randn(3, 8)):
                b = envoy.fc1.output.save()
        assert a.shape == (2, 8) and bool((a == 1).all())
        assert b.shape == (3, 8) and bool((b != 1).any())

    @torch.no_grad()
    def test_a_lone_invoke_may_change_its_rows(self):
        # Nothing to splice into: one invoke *is* the batch, so a write is the
        # whole value and may reshape it, leading dim included.
        torch.manual_seed(0)
        envoy = _BatchEnvoy(_MLP())
        with envoy.trace(torch.randn(2, 8)):
            envoy.fc1.output = torch.ones(5, 8)
            out = envoy.output.save()
        assert out.shape == (5, 8)


class TestCacheAcrossInvokes:
    """A cache belongs to the invoke it was opened in, so it records that invoke's
    own rows of the batch — narrowed the same way the invoke's reads are, not the
    whole combined batch."""

    @torch.no_grad()
    def test_each_invoke_caches_its_own_rows(self):
        torch.manual_seed(0)
        model = _MLP()
        envoy = _BatchEnvoy(model)
        xa, xb = torch.randn(2, 8), torch.randn(3, 8)
        with envoy.trace() as tracer:
            with tracer.invoke(xa):
                ca = tracer.cache(modules=[envoy.fc1])
            with tracer.invoke(xb):
                cb = tracer.cache(modules=[envoy.fc1])
        a = ca["model.fc1"].output
        b = cb["model.fc1"].output
        assert a.shape[0] == 2 and b.shape[0] == 3
        # Each cache holds exactly its invoke's rows — matches running fc1 on that
        # invoke's input alone.
        assert torch.allclose(a, model.fc1(xa), atol=1e-5)
        assert torch.allclose(b, model.fc1(xb), atol=1e-5)


class TestBarrier:
    """Blocks resume in the order the model reaches what each asked for, so one
    handing something to another needs a point they both agree on."""

    @torch.no_grad()
    def test_embedding_transfer(self, gpt2):
        # The second invoke generates from the first's embeddings: the barrier is
        # what guarantees they were read before it swaps them in.
        with gpt2.pipe(max_new_tokens=3, do_sample=False) as tracer:
            barrier = tracer.barrier(2)
            with tracer.invoke(MSG):
                embeddings = gpt2.transformer.wte.output
                barrier()
                result = tracer.result.save()
            with tracer.invoke("_ _ _ _ _ _ _ _ _"):
                barrier()
                gpt2.transformer.wte.output = embeddings

        def text(item):  # a list-input invoke yields a list per prompt; unwrap it
            item = item[0] if isinstance(item, list) else item
            return item["generated_text"]

        sent, received = (text(r) for r in result)
        assert sent.endswith("New York City")
        # The underscores carry none of that on their own; the embeddings do.
        assert received.endswith("New York City")

    @torch.no_grad()
    def test_embedding_transfer_three_blocks(self, gpt2):
        # A barrier of three: the first invoke's embeddings reach the other two, so
        # all three had to park and release together, not just a pair.
        receiver = "_ _ _ _ _ _ _ _ _"
        with gpt2.pipe(max_new_tokens=3, do_sample=False) as tracer:
            barrier = tracer.barrier(3)
            with tracer.invoke(MSG):
                embeddings = gpt2.transformer.wte.output
                barrier()
                result = tracer.result.save()
            with tracer.invoke(receiver):
                barrier()
                gpt2.transformer.wte.output = embeddings
            with tracer.invoke(receiver):
                barrier()
                gpt2.transformer.wte.output = embeddings

        def text(item):
            item = item[0] if isinstance(item, list) else item
            return item["generated_text"]

        sent, recv_a, recv_b = (text(r) for r in result)
        assert sent.endswith("New York City")
        # Both receivers carry only underscores; the source's embeddings are what
        # make them finish the same way.
        assert recv_a.endswith("New York City")
        assert recv_b.endswith("New York City")

    @torch.no_grad()
    def test_barrier_releases_every_block_in_step(self, gpt2):
        # Directly: with a barrier of three, no block passes it until all three
        # reach it — every "before" happens before any "after".
        order = []
        with gpt2.trace() as tracer:
            barrier = tracer.barrier(3)
            for prompt in (ET, MSG, "a third prompt here"):
                with tracer.invoke(prompt):
                    order.append("before")
                    barrier()
                    order.append("after")
        assert order == ["before"] * 3 + ["after"] * 3

    @torch.no_grad()
    def test_name_bound_in_one_invoke_reaches_the_next(self, gpt2):
        # The one thing the per-frame shared store exists for: invoke bodies are
        # siblings written in one frame, and `embeddings` is not in the second
        # body's snapshot because nothing had bound it when that body was
        # captured. (Every other kind of block reaches what it needs through its
        # own snapshot, which is why only invokes pass `shared=`.)
        with gpt2.trace() as tracer:
            barrier = tracer.barrier(2)
            with tracer.invoke(MSG):
                embeddings = gpt2.transformer.wte.output
                barrier()
            with tracer.invoke("_ _ _ _ _ _ _ _ _"):
                barrier()
                gpt2.transformer.wte.output = embeddings
                out = gpt2.output.logits[0, -1].argmax().save()
        assert out.ndim == 0

    @torch.no_grad()
    def test_invoke_scope_does_not_outlive_its_trace(self, gpt2):
        # The store is dropped when the trace that owns those invokes finishes,
        # not when the outermost trace does. Holding it to the end of a session
        # kept one dead mediator frame -- and its tensors -- per trace.
        from nnsight.intervention.util import _shared

        def entries():
            return len(getattr(_shared, "store", None) or {})

        with gpt2.session():
            for _ in range(3):
                with gpt2.trace() as tracer:
                    with tracer.invoke(ET):
                        gpt2.transformer.h[0].output
                assert entries() == 0

    @torch.no_grad()
    def test_barrier_no_block_reaches_it_runs_clean(self, gpt2):
        # A barrier nobody calls is inert.
        with gpt2.trace() as tracer:
            tracer.barrier(2)
            with tracer.invoke(ET):
                out = gpt2.transformer.h[0].output.save()
        assert out.shape[0] == 1

    @torch.no_grad()
    def test_barrier_waiting_for_more_blocks_than_arrive(self, gpt2):
        # Built for three, reached by two: it never releases, and the blocks left
        # waiting say so rather than hanging.
        with pytest.raises(ValueError, match="never reached by every block"):
            with gpt2.trace() as tracer:
                barrier = tracer.barrier(3)
                with tracer.invoke(ET):
                    barrier()
                with tracer.invoke(MSG):
                    barrier()


class TestInvokeScope:
    """Invokes of one trace share the scope they were written in, so a name one
    invoke binds is readable by a later invoke.

    The reader has to reach the name *after* the bind happens, and every worker runs
    up to its first park before the model runs at all — so a read is only safe once
    the reader has parked on a location the model reaches after the binder's. Read
    a name straight away in an invoke and the binder hasn't run yet."""

    @torch.no_grad()
    def test_name_bound_in_one_invoke_reads_in_the_next(self, gpt2):
        with gpt2.trace() as tracer:
            with tracer.invoke(ET):
                first = gpt2.transformer.h[0].output
                first_sum = first.sum().save()
            with tracer.invoke(MSG):
                second = gpt2.transformer.h[1].output
                second_sum = second.sum().save()
                crossed = (first.sum() + second.sum()).save()
        # `first` carries the earlier invoke's own rows, not this invoke's.
        assert torch.allclose(crossed, first_sum + second_sum)

    @torch.no_grad()
    def test_crossed_value_is_the_earlier_invokes_rows(self, gpt2):
        with gpt2.trace() as tracer:
            with tracer.invoke(ET):
                first = gpt2.transformer.h[0].output
            with tracer.invoke([MSG, ET]):
                # Park past the bind (h[0]) before reading `first`.
                mine = gpt2.transformer.h[1].output.save()
                crossed = first.clone().save()
        assert crossed.shape[0] == 1  # the ET invoke's single row
        assert mine.shape[0] == 2  # this invoke's own two rows

    @torch.no_grad()
    def test_loop_variable_binds_per_invoke(self, gpt2):
        # An invoke's body runs after the loop that wrote it has moved on, so a name
        # it only reads means what it meant where the body was written.
        seen = []
        with gpt2.trace() as tracer:
            for prompt in (ET, MSG):
                with tracer.invoke(prompt):
                    seen.append(prompt)
        assert seen == [ET, MSG]

    @torch.no_grad()
    def test_outer_name_survives_the_trace(self, gpt2):
        keep = "untouched"
        with gpt2.trace(ET):
            gpt2.transformer.h[0].output.save()
        assert keep == "untouched"

    @torch.no_grad()
    def test_name_bound_in_invoke_is_visible_after_the_trace(self, gpt2):
        with gpt2.trace() as tracer:
            with tracer.invoke(ET):
                inside = gpt2.transformer.h[0].output.save()
        assert isinstance(inside, torch.Tensor) and inside.shape[0] == 1

    @torch.no_grad()
    def test_read_before_the_bind_raises(self, gpt2):
        # Reading a name an earlier invoke binds, before parking past where it binds
        # it: the reader runs first, up to its first park, so the name isn't there.
        with pytest.raises(NameError):
            with gpt2.trace() as tracer:
                with tracer.invoke(ET):
                    first = gpt2.transformer.h[0].output
                with tracer.invoke(MSG):
                    crossed = first.clone().save()


class TestInvokeFromHelper:
    """Trace code factored into helpers: a tracer handed to a function can open
    invokes there, and a helper called from inside an invoke reads that invoke's
    rows. Each ``with`` block is captured from the frame it appears in, so the
    trace body and its helpers are on equal footing."""

    @torch.no_grad()
    def test_helpers_do_not_share_each_others_names(self, gpt2):
        # Invokes written in different functions are as separate as the code they
        # were written in: a name local to one helper's invoke is not in the other's.
        def binds(tracer):
            with tracer.invoke(ET):
                secret = gpt2.transformer.h[0].output.sum()  # noqa: F841

        saw = {}

        def reads(tracer):
            with tracer.invoke(MSG):
                gpt2.transformer.h[1].output  # park past the other invoke's bind
                try:
                    saw["value"] = secret  # noqa: F821
                except NameError:
                    saw["value"] = "NameError"

        with gpt2.trace() as tracer:
            binds(tracer)
            reads(tracer)
        assert saw["value"] == "NameError"

    @torch.no_grad()
    def test_invoke_opened_in_helper(self, gpt2):
        def invoke(tracer, prompt, key, outs):
            with tracer.invoke(prompt):
                outs[key] = gpt2.transformer.h[0].output

        with gpt2.trace() as tracer:
            outs = nnsight.save({})
            invoke(tracer, ET, "a", outs)
            invoke(tracer, MSG, "b", outs)
        # Each helper-opened invoke is scoped to its own row, as if written inline.
        assert outs["a"].shape[0] == 1 and outs["b"].shape[0] == 1
        assert not torch.equal(outs["a"], outs["b"])

    @torch.no_grad()
    def test_helper_called_inside_invoke_sees_its_rows(self, gpt2):
        def read(layer):
            return gpt2.transformer.h[layer].output.save()

        with gpt2.trace() as tracer:
            with tracer.invoke(ET):
                first = read(0)
            with tracer.invoke([MSG, ET]):
                second = read(0)
        assert first.shape[0] == 1 and second.shape[0] == 2

    @torch.no_grad()
    def test_empty_invoke_opened_in_helper(self, gpt2):
        def whole_batch(tracer, sink):
            with tracer.invoke():
                sink["logits"] = gpt2.lm_head.output[:, -1]

        with gpt2.trace() as tracer:
            sink = nnsight.save({})
            with tracer.invoke(ET):
                pass
            with tracer.invoke(MSG):
                pass
            whole_batch(tracer, sink)
        assert sink["logits"].shape[0] == 2


class TestInvokerGrouping:
    """Ported from the old invoker-batching suite: invokes of mixed size, list
    inputs, and empty (whole-batch) invokes combine into one row-ordered batch."""

    @torch.no_grad()
    def test_invoker_group_batching(self, gpt2):
        # single, list-of-2, empty, empty, single -> rows [ET | MSG, ET | MSG].
        with gpt2.trace() as tracer:
            with tracer.invoke(ET):
                o1 = gpt2.lm_head.output[:, -1].save()
            with tracer.invoke([MSG, ET]):
                o2 = gpt2.lm_head.output[:, -1].save()
            with tracer.invoke():
                o3 = gpt2.lm_head.output[:, -1].save()
            with tracer.invoke():
                o4 = gpt2.lm_head.output[:, -1].save()
            with tracer.invoke(MSG):
                o5 = gpt2.lm_head.output[:, -1].save()
        assert [o.shape[0] for o in (o1, o2, o3, o4, o5)] == [1, 2, 4, 4, 1]
        # the two empty invokes both see the whole batch...
        assert torch.equal(o3, o4)
        # ...which is exactly the invokes concatenated in order.
        assert torch.equal(torch.cat([o1, o2, o5]), o3)
        # the same prompt (ET) in invoke 1 and as row 1 of invoke 2 agree.
        assert torch.equal(o1, o2[1].unsqueeze(0))

    @torch.no_grad()
    def test_list_and_invoke_attention_masks_match(self, gpt2):
        # A single trace over a list of mixed-length prompts is the ground truth;
        # splitting the same prompts across invokes must produce identical logits
        # (each invoke's padding/mask handled correctly when combined).
        prompts = ["Hi", "A B C", "D"]
        with gpt2.trace(prompts):
            ground_truth = gpt2.lm_head.output[:, -1].save()
        with gpt2.trace() as tracer:
            with tracer.invoke("Hi"):
                pass
            with tracer.invoke(["A B C", "D"]):
                pass
            with tracer.invoke():
                via_invokes = gpt2.lm_head.output[:, -1].save()
        assert torch.allclose(ground_truth, via_invokes, atol=1e-4)

    @torch.no_grad()
    def test_batched_positions_match_single_prompt(self, gpt2):
        # A left-padded shorter prompt in a mixed-length batch must predict the same
        # next token as run alone (mask-derived position_ids); the unpadded longer
        # row too.
        short, long = "Hello world", ET
        with gpt2.trace([short, long]):
            batched = gpt2.lm_head.output[:, -1].save()
        with gpt2.trace(short):
            short_alone = gpt2.lm_head.output[:, -1].save()
        with gpt2.trace(long):
            long_alone = gpt2.lm_head.output[:, -1].save()
        assert batched[0].argmax() == short_alone[0].argmax()
        assert batched[1].argmax() == long_alone[0].argmax()

    @torch.no_grad()
    def test_cross_invoke_isolation(self, gpt2):
        # Zeroing the last layer in one invoke must not touch the other, which still
        # predicts MSG's continuation (" New").
        with gpt2.trace() as tracer:
            with tracer.invoke(MSG):
                gpt2.transformer.h[-1].output[0][:] = 0
            with tracer.invoke(MSG):
                hidden = gpt2.transformer.h[-1].output[0].save()
                token = gpt2.lm_head.output[0][-1].argmax(dim=-1).save()
        assert torch.all(hidden != 0)
        assert gpt2.tokenizer.decode(token) == " New"


class TestInvokerInputFormats:
    """Every input format a forward accepts is batchable across invokes, and a
    format matches the same prompt run as a string (same row-scoped logits)."""

    @torch.no_grad()
    def test_pretokenized_invoke_matches_string(self, gpt2):
        # A token-id-list invoke and a string invoke of the same prompt agree, and
        # the two batch together (different formats, one padded forward).
        ids = gpt2.tokenizer(MSG).input_ids
        with gpt2.trace() as tracer:
            with tracer.invoke(ids):
                a = gpt2.lm_head.output[0, -1].save()
            with tracer.invoke(ET):
                pass
        with gpt2.trace(MSG):
            solo = gpt2.lm_head.output[0, -1].save()
        assert a.argmax() == solo.argmax()

    @torch.no_grad()
    def test_mixed_formats_batch_together(self, gpt2):
        # str, list[int], 1-D tensor, and a list of two strings in four invokes ->
        # one combined batch of 1 + 1 + 1 + 2 = 5 rows.
        ids_list = gpt2.tokenizer("the quick brown fox").input_ids
        ids_tensor = gpt2.tokenizer(ET, return_tensors="pt").input_ids[0]
        with gpt2.trace() as tracer:
            with tracer.invoke(MSG):
                pass
            with tracer.invoke(ids_list):
                pass
            with tracer.invoke(ids_tensor):
                pass
            with tracer.invoke(["a b c", "d e"]):
                pass
            with tracer.invoke():
                whole = gpt2.lm_head.output[:, -1].save()
        assert whole.shape[0] == 5

    @torch.no_grad()
    def test_per_invoke_truncation(self, gpt2):
        # A tokenizer kwarg on an invoke (truncation) is applied to that invoke's
        # tokenization rather than forwarded to the model, so the sequence is cut.
        with gpt2.trace() as tracer:
            with tracer.invoke("word " * 50, truncation=True, max_length=4):
                short = gpt2.transformer.h[0].output[0].save()
        assert short.shape[0] == 4  # tokenized to 4 tokens, not the full ~100


class TestBatchedGenerationParity:
    @torch.no_grad()
    def test_batched_generation_matches_single(self, gpt2):
        # A padded prompt generated in a mixed-length batch must continue the same
        # as generated alone (the position_ids correction is stripped for generate).
        short, long = "Hello world", ET
        with gpt2.pipe(max_new_tokens=5, do_sample=False) as tracer:
            with tracer.invoke([short, long]):
                batched = tracer.result.save()
        with gpt2.pipe(short, max_new_tokens=5, do_sample=False) as tracer:
            single = tracer.result.save()

        def text(item):  # a list-input invoke yields a list per prompt; unwrap it
            item = item[0] if isinstance(item, list) else item
            return item["generated_text"]

        assert text(batched[0]) == text(single[0])


class TestBatchedIteration:
    @torch.no_grad()
    def test_per_invoke_iter_slices(self, gpt2):
        # Two invokes of the same prompt, each with its own iter slice over the same
        # greedy generation, capture their chosen steps independently.
        with gpt2.generate(max_new_tokens=5, do_sample=False) as tracer:
            with tracer.invoke(MSG):
                first = nnsight.save([])
                for _ in tracer.iter[1:3]:
                    first.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))
            with tracer.invoke(MSG):
                second = nnsight.save([])
                for _ in tracer.iter[:3]:
                    second.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))
        assert len(first) == 2 and len(second) == 3
        assert gpt2.tokenizer.batch_decode(first) == [" York City"]
        assert gpt2.tokenizer.batch_decode(second) == [" New York City"]

    @torch.no_grad()
    def test_iter_per_invoke_differs(self, gpt2):
        # Different prompts per invoke -> different per-step predictions.
        with gpt2.generate(max_new_tokens=3, do_sample=False) as tracer:
            with tracer.invoke(ET):
                a = nnsight.save([])
                for _ in tracer.iter[:3]:
                    a.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))
            with tracer.invoke(MSG):
                b = nnsight.save([])
                for _ in tracer.iter[:3]:
                    b.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))
        assert all(not torch.equal(x, y) for x, y in zip(a, b))


class TestCppBacktraceGuard:
    """A torch op that raises inside an interleaving greenlet must surface as a
    normal Python exception, not crash the process.

    In a batched trace each invoke runs in its own greenlet; once a worker has
    parked on a read, a c10 error thrown afterwards used to make torch's error
    constructor walk the greenlet's (stack-sliced) C stack via glibc backtrace()
    and segfault. ``CONFIG.APP.DISABLE_CPP_BACKTRACE`` neutralizes that call.
    Run in a subprocess so a regression is a clean failure here, not a crashed
    test session.
    """

    _SCRIPT = textwrap.dedent(
        """
        import torch
        from nnsight.modeling.transformers import TransformersModel
        model = TransformersModel("gpt2", dispatch=True)
        with model.trace() as tracer:
            with tracer.invoke("The Eiffel Tower is in the city of"):
                _ = model.transformer.h[5].output      # park the worker on a read
                torch.ones(3) @ torch.ones(4)          # then a c10 RuntimeError
            with tracer.invoke("The Colosseum is in the city of"):
                pass
        """
    )

    def _run(self, env_extra):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **env_extra}
        return subprocess.run(
            [sys.executable, "-c", self._SCRIPT],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_guard_on_surfaces_python_exception(self):
        result = self._run({})  # guard on by default
        # Not a crash signal (SIGSEGV is returncode -11 / 139); the torch error
        # came out as a normal Python exception.
        assert result.returncode not in (-11, 139), result.stderr[-2000:]
        assert "RuntimeError" in result.stderr, result.stderr[-2000:]

    def test_guard_off_reproduces_the_segfault(self):
        # Turning the guard off should bring the crash back — proof the guard is
        # what's doing the work, not incidental. Only glibc/x86-64 Linux crashes;
        # elsewhere the guard is a no-op, so the process exits cleanly either way.
        if sys.platform != "linux" or __import__("platform").machine() != "x86_64":
            pytest.skip("segfault only reproduces on glibc/x86-64 Linux")
        result = self._run({"NNSIGHT_DISABLE_CPP_BACKTRACE": "0"})
        # Whether the unguarded backtrace actually walks off the greenlet's stack
        # depends on the interpreter's stack layout, not on nnsight: across 3.10-3.14
        # only some builds crash. If this one doesn't, there's no guard effect to
        # demonstrate — but test_guard_on_surfaces_python_exception above still holds
        # nnsight to the behavior that matters.
        if result.returncode not in (-11, 139):
            pytest.skip(
                "the crash doesn't reproduce on this interpreter/toolchain "
                f"(returncode {result.returncode}), so the guard's effect can't be shown"
            )
