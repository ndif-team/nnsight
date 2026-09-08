"""Tensor parallelism: sharded activations read and edit as whole tensors.

Under TP a linear layer's value at any one rank is only that rank's shard. The
port gathers those shards before a worker sees the value and re-shards whatever
the worker leaves before vLLM's own forward carries on. These tests pin that down
by running the identical trace on an unsharded reference (``vllm_qwen_ref``, one
rank) and on the sharded engine (``vllm_qwen_tp``) and comparing **element-wise**:
the whole a worker sees is the tensor the single-rank run produced, not a
permutation of its values.

Element-wise is the point. Row-parallel layers (``o_proj``, ``down_proj``) split
the contraction and all-reduce their output, so their whole is already laid out
like the single-rank run's. Column-parallel layers here are *fused* (``qkv_proj``
packs Q/K/V, ``gate_up_proj`` packs gate/up) and each rank holds a slice of every
packed part, so the ranks' concatenation is ``[q0 k0 v0 | q1 k1 v1]`` — all the
same values, grouped differently. `VLLMFragments` un-interleaves that back to
``[q | k | v]`` on the way in and re-interleaves on the way out, so a recipe that
slices by head or by ``chunk(2, -1)`` means the same thing at every ``tp_size``.
Comparing a column-parallel read as a per-row multiset passes on either layout,
which is exactly why these tests do not.

`TestFusedLayout` checks the reordering arithmetic on its own, including the
grouped-query case where there are fewer KV heads than ranks and vLLM replicates
K and V across a group — which no checkpoint small enough to run here reaches.

The whole module is skipped unless the machine has >=2 GPUs.
"""

import pytest
import torch

pytest.importorskip("vllm")

from nnsight.modeling.vllm import fragments

LAYER = 5
COLUMN_PARALLEL = ["self_attn.qkv_proj", "mlp.gate_up_proj"]
ROW_PARALLEL = ["self_attn.o_proj", "mlp.down_proj"]
ALL_PARALLEL = COLUMN_PARALLEL + ROW_PARALLEL


def _submodule(model, path):
    obj = model.model.layers[LAYER]
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _read(model, path, prompt):
    with model.trace(prompt, temperature=0.0, top_p=1):
        hs = _submodule(model, path).output[0].save()
        logits = model.logits.save()
    return hs, logits


def _zero_tail(model, path, prompt):
    """Zero the upper half of a layer's gathered output."""
    with model.trace(prompt, temperature=0.0, top_p=1):
        out = _submodule(model, path).output
        value = out[0].clone()
        value[:, value.shape[-1] // 2 :] = 0
        _submodule(model, path).output = (value, *out[1:])
        logits = model.logits.save()
    return logits


def _zero_all(model, path, prompt):
    """Zero the whole of a layer's gathered output, layout-independent."""
    with model.trace(prompt, temperature=0.0, top_p=1):
        out = _submodule(model, path).output
        value = torch.zeros_like(out[0])
        _submodule(model, path).output = (value, *out[1:])
        logits = model.logits.save()
    return logits


def _mlp_out_with_up_zeroed(model, prompt):
    """The MLP's output once the upper half of its fused ``gate_up`` is zeroed."""
    with model.trace(prompt, temperature=0.0, top_p=1):
        gate_up = _submodule(model, "mlp.gate_up_proj")
        out = gate_up.output
        value = out[0].clone()
        value[:, value.shape[-1] // 2 :] = 0
        gate_up.output = (value, *out[1:])
        mlp_out = _submodule(model, "mlp").output.clone().save()
    return mlp_out


def _min_row_cosine(a, b):
    """The least cosine similarity between corresponding rows of ``a`` and ``b``.

    Robust to the few-percent drift bf16 accumulates between one- and many-rank
    runs, where an absolute tolerance is not.
    """
    return torch.nn.functional.cosine_similarity(
        a.float(), b.float(), dim=-1
    ).min().item()


class TestShardedRead:
    """A gathered value carries the whole layer's data, not a rank's shard."""

    @pytest.mark.parametrize("path", ROW_PARALLEL)
    @torch.no_grad()
    def test_row_parallel_matches_reference(
        self, vllm_qwen_ref, vllm_qwen_tp, path, ET_prompt
    ):
        ref_hs, ref_logits = _read(vllm_qwen_ref, path, ET_prompt)
        tp_hs, tp_logits = _read(vllm_qwen_tp, path, ET_prompt)

        assert tp_hs.shape == ref_hs.shape
        assert tp_logits.argmax(dim=-1).item() == ref_logits.argmax(dim=-1).item()
        assert _min_row_cosine(tp_hs, ref_hs) > 0.99

    @pytest.mark.parametrize("path", COLUMN_PARALLEL)
    @torch.no_grad()
    def test_column_parallel_matches_reference(
        self, vllm_qwen_ref, vllm_qwen_tp, path, ET_prompt
    ):
        ref_hs, ref_logits = _read(vllm_qwen_ref, path, ET_prompt)
        tp_hs, tp_logits = _read(vllm_qwen_tp, path, ET_prompt)

        # Column-by-column, not as a multiset: both layers are fused, and a
        # gather left in rank order holds every value this reference holds while
        # putting Q where K belongs.
        assert tp_hs.shape == ref_hs.shape
        assert tp_logits.argmax(dim=-1).item() == ref_logits.argmax(dim=-1).item()
        assert _min_row_cosine(tp_hs, ref_hs) > 0.99

    @pytest.mark.parametrize("path", ROW_PARALLEL)
    @torch.no_grad()
    def test_row_parallel_input_is_gathered(
        self, vllm_qwen_ref, vllm_qwen_tp, path, ET_prompt
    ):
        # A row-parallel layer takes its input already split across ranks; reading
        # it must gather to the same whole the single-rank run sees.
        with vllm_qwen_ref.trace(ET_prompt, temperature=0.0, top_p=1):
            ref_in = _submodule(vllm_qwen_ref, path).input.save()
        with vllm_qwen_tp.trace(ET_prompt, temperature=0.0, top_p=1):
            tp_in = _submodule(vllm_qwen_tp, path).input.save()

        assert tp_in.shape == ref_in.shape


def _greedy_tokens(model, prompt, n):
    with model.trace(prompt, temperature=0.0, top_p=1.0, max_tokens=n) as tracer:
        toks = list().save()
        for _ in tracer.iter[:n]:
            toks.append(model.logits.argmax(dim=-1))
    return [t.item() for t in toks]


class TestShardedRequests:
    """Request tracking holds up while the model is sharded across ranks."""

    @torch.no_grad()
    def test_generation_matches_reference(
        self, vllm_qwen_ref, vllm_qwen_tp, MSG_prompt
    ):
        assert _greedy_tokens(vllm_qwen_tp, MSG_prompt, 4) == _greedy_tokens(
            vllm_qwen_ref, MSG_prompt, 4
        )

    @torch.no_grad()
    def test_batched_requests_keep_their_spans(
        self, vllm_qwen_ref, vllm_qwen_tp, ET_prompt, MSG_prompt
    ):
        et_n = len(vllm_qwen_tp.tokenizer.encode(ET_prompt))
        msg_n = len(vllm_qwen_tp.tokenizer.encode(MSG_prompt))

        with vllm_qwen_tp.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                et_hs = vllm_qwen_tp.model.layers[0].self_attn.qkv_proj.input.save()
                et_logits = vllm_qwen_tp.logits.save()
            with tracer.invoke(MSG_prompt):
                msg_hs = vllm_qwen_tp.model.layers[0].self_attn.qkv_proj.input.save()
                msg_logits = vllm_qwen_tp.logits.save()

        # Each sharded request is still narrowed to exactly its own tokens.
        assert et_hs.shape[0] == et_n
        assert msg_hs.shape[0] == msg_n

        with vllm_qwen_ref.trace(ET_prompt, temperature=0.0, top_p=1):
            ref_et = vllm_qwen_ref.logits.save()
        with vllm_qwen_ref.trace(MSG_prompt, temperature=0.0, top_p=1):
            ref_msg = vllm_qwen_ref.logits.save()

        assert et_logits.argmax(dim=-1).item() == ref_et.argmax(dim=-1).item()
        assert msg_logits.argmax(dim=-1).item() == ref_msg.argmax(dim=-1).item()

    @torch.no_grad()
    def test_intervention_during_sharded_generation(self, vllm_qwen_tp, MSG_prompt):
        with vllm_qwen_tp.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=4
        ) as tracer:
            clean = list().save()
            for _ in tracer.iter[:4]:
                clean.append(vllm_qwen_tp.logits)

        with vllm_qwen_tp.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=4
        ) as tracer:
            edited = list().save()
            for it in tracer.iter[:4]:
                if it == 1:
                    out = vllm_qwen_tp.model.layers[-2].mlp.down_proj.output
                    vllm_qwen_tp.model.layers[-2].mlp.down_proj.output = (
                        torch.zeros_like(out[0]),
                        *out[1:],
                    )
                edited.append(vllm_qwen_tp.logits)

        # A sharded edit at one decode step changes that step's logits.
        assert not torch.allclose(clean[1].float(), edited[1].float())


class TestShardedEdit:
    """An edit to a gathered value is re-sharded back into vLLM's forward."""

    @pytest.mark.parametrize("path", ALL_PARALLEL)
    @torch.no_grad()
    def test_edit_lands(self, vllm_qwen_tp, path, ET_prompt):
        _, clean_logits = _read(vllm_qwen_tp, path, ET_prompt)
        edited_logits = _zero_tail(vllm_qwen_tp, path, ET_prompt)

        # Editing the gathered value and re-sharding it reaches vLLM's forward:
        # the prediction moves off the clean one.
        assert not torch.allclose(edited_logits.float(), clean_logits.float())

    @pytest.mark.parametrize("path", ALL_PARALLEL)
    @torch.no_grad()
    def test_whole_edit_matches_reference(
        self, vllm_qwen_ref, vllm_qwen_tp, path, ET_prompt
    ):
        # Zeroing the entire output is the same logical edit whatever the layout,
        # so the re-sharded sharded run and the unsharded run must land together.
        ref_logits = _zero_all(vllm_qwen_ref, path, ET_prompt)
        tp_logits = _zero_all(vllm_qwen_tp, path, ET_prompt)

        assert tp_logits.argmax(dim=-1).item() == ref_logits.argmax(dim=-1).item()

    @torch.no_grad()
    def test_half_edit_means_the_same_half(
        self, vllm_qwen_ref, vllm_qwen_tp, ET_prompt
    ):
        # `gate_up_proj` packs `[gate | up]` and the activation is
        # `silu(gate) * up`, so zeroing the upper half of the gathered value
        # zeroes the MLP outright — but only if that half really is `up`. In
        # rank order it is rank 1's gate *and* up, and the MLP goes on producing
        # something. This is the edit that `_zero_all` cannot catch.
        ref_out = _mlp_out_with_up_zeroed(vllm_qwen_ref, ET_prompt)
        tp_out = _mlp_out_with_up_zeroed(vllm_qwen_tp, ET_prompt)

        assert ref_out.abs().max().item() == 0.0, "reference: does down_proj bias?"
        assert tp_out.abs().max().item() == 0.0


class TestAdHocCall:
    """An ad-hoc call on a sharded module takes and returns whole tensors.

    ``ParallelEnvoy.__call__`` is the only consumer of ``Fragments.split`` and
    ``Fragments.whole`` outside the interleaver, so these are the tests that
    keep that callsite on the current API — it once drifted onto a deleted
    method name and shipped ``whole()``'s ``(value, undo)`` record to the
    caller. Both styles are exercised because they break differently: a
    row-parallel call splits the caller's whole input on the way in (the drift
    was an ``AttributeError``), a column-parallel call reassembles the sharded
    output on the way out (the drift was a silent wrong value).
    """

    @torch.no_grad()
    def test_row_parallel_call_takes_and_returns_the_whole(
        self, vllm_qwen_ref, vllm_qwen_tp, ET_prompt
    ):
        path = "mlp.down_proj"

        with vllm_qwen_tp.trace(ET_prompt, temperature=0.0, top_p=1):
            module = _submodule(vllm_qwen_tp, path)
            hidden = module.input  # gathered whole; the caller holds the real thing
            expected = module.output[0].save()
            result = module(hidden)
            # The module's own (output, bias) pair, not whole()'s (value, undo).
            assert torch.is_tensor(result[0]), f"ad-hoc call returned {type(result[0])}"
            adhoc = result[0].save()

        assert adhoc.shape == expected.shape
        assert _min_row_cosine(adhoc, expected) > 0.99

        with vllm_qwen_ref.trace(ET_prompt, temperature=0.0, top_p=1):
            module = _submodule(vllm_qwen_ref, path)
            ref = module(module.input)[0].save()

        # A row-parallel output is all-reduced, so its layout matches the
        # single-rank run's and the values compare directly.
        assert adhoc.shape == ref.shape
        assert _min_row_cosine(adhoc, ref) > 0.99

    @torch.no_grad()
    def test_column_parallel_call_returns_the_whole(
        self, vllm_qwen_ref, vllm_qwen_tp, ET_prompt
    ):
        path = "mlp.gate_up_proj"

        with vllm_qwen_tp.trace(ET_prompt, temperature=0.0, top_p=1):
            module = _submodule(vllm_qwen_tp, path)
            hidden = module.input  # replicated, already whole
            expected = module.output[0].save()
            result = module(hidden)
            assert torch.is_tensor(result[0]), f"ad-hoc call returned {type(result[0])}"
            adhoc = result[0].save()

        # Same engine, same gather: the reassembled ad-hoc output lays out
        # exactly as the traced read of the same location.
        assert adhoc.shape == expected.shape
        assert _min_row_cosine(adhoc, expected) > 0.99

        with vllm_qwen_ref.trace(ET_prompt, temperature=0.0, top_p=1):
            module = _submodule(vllm_qwen_ref, path)
            ref = module(module.input)[0].save()

        # And the reassembled whole is the single-rank layer's own output,
        # column for column (as TestShardedRead checks for a traced read).
        assert adhoc.shape == ref.shape
        assert _min_row_cosine(adhoc, ref) > 0.99


class TestEveryRankWindsUp:
    """Cleanup is per rank, not just the one whose values go home.

    Every rank runs the block, so every rank holds a worker, its greenlet, and
    whatever that greenlet captured. Only rank 0's values are reported, and the
    collect used to return early on the others — leaving all of it in place for
    the life of the engine. `nnsight_request_count` is the gauge, and the reason
    the leak survived is that the one test using it looked only at `counts[0]`.
    """

    @torch.no_grad()
    def test_no_worker_is_left_on_any_rank(self, vllm_qwen_tp, ET_prompt):
        model = vllm_qwen_tp
        engine = model.vllm_entrypoint.llm_engine

        for _ in range(3):
            with model.trace(ET_prompt, temperature=0.0, top_p=1):
                hidden = model.model.layers[LAYER].output[0].save()

        counts = engine.collective_rpc("nnsight_request_count")
        assert len(counts) > 1, "expected more than one rank"
        assert counts == [0] * len(counts), f"workers left behind per rank: {counts}"

    @torch.no_grad()
    def test_an_installed_block_leaves_nothing_either(self, vllm_qwen_tp, ET_prompt):
        model = vllm_qwen_tp
        engine = model.vllm_entrypoint.llm_engine

        with model.edit() as (tracer, edit):
            hidden = model.model.layers[LAYER].output[0].save()
        try:
            for _ in range(3):
                model.generate([ET_prompt], max_tokens=2, temperature=0.0,
                               ignore_eos=True)

            counts = engine.collective_rpc("nnsight_request_count")
            assert counts == [0] * len(counts), f"workers left behind: {counts}"
        finally:
            edit.clear()


def _fused_reference(widths, replicas, tp_size):
    """A single-rank ``[q | k | v]``, and the all-gather vLLM builds out of it.

    ``widths`` are one rank's, so a projection replicated across ``replicas``
    adjacent ranks is only ``tp_size // replicas`` shards wide in the whole.
    Which shard a rank holds is vLLM's own ``tp_rank // num_kv_head_replicas``.
    """
    sizes = [width * (tp_size // every) for width, every in zip(widths, replicas)]
    whole = torch.arange(3 * sum(sizes), dtype=torch.float32).reshape(3, -1)

    blocks, offset = [], 0
    for size in sizes:
        blocks.append(whole[:, offset : offset + size])
        offset += size

    shards = [
        block[:, (rank // every) * width : (rank // every) * width + width]
        for rank in range(tp_size)
        for block, width, every in zip(blocks, widths, replicas)
    ]
    return whole, torch.cat(shards, dim=-1)


LAYOUTS = [
    ([1024, 256, 256], [1, 1, 1], 2),      # Llama-3.2-1B qkv: 32 q / 8 kv heads
    ([4096, 4096], [1, 1], 2),             # its gate_up
    ([448, 64, 64], [1, 1, 1], 2),         # Qwen2.5-0.5B qkv: 14 q / 2 kv heads
    ([128, 64, 64], [1, 2, 2], 4),         # 8 q / 2 kv at tp=4: K and V doubled
    ([512, 128, 128], [1, 8, 8], 8),       # multi-query: one KV head on 8 ranks
    ([64, 128, 32, 32], [1, 1, 1, 1], 4),  # a four-way merged column
]


class TestFusedLayout:
    """The reordering on its own, without an engine.

    Its hard case is grouped-query attention with fewer KV heads than ranks:
    vLLM replicates K and V across a group of adjacent ranks, so the gather
    carries each of them once per rank in the group and a plain un-interleave
    would hand back a tensor several KV heads too wide. No checkpoint small
    enough for these fixtures reaches it — Qwen2.5-0.5B has 14 query heads and
    so cannot shard past 2 — hence the arithmetic is checked against a model of
    vLLM's own shard assignment instead.
    """

    @pytest.mark.parametrize("widths, replicas, tp_size", LAYOUTS)
    def test_unfuse_undoes_the_gather(self, widths, replicas, tp_size):
        whole, gathered = _fused_reference(widths, replicas, tp_size)

        assert torch.equal(
            fragments._unfuse(gathered, widths, replicas, tp_size), whole
        )

    @pytest.mark.parametrize("widths, replicas, tp_size", LAYOUTS)
    def test_fuse_gives_each_rank_back_its_own_piece(self, widths, replicas, tp_size):
        whole, gathered = _fused_reference(widths, replicas, tp_size)
        per_rank = sum(widths)

        for rank in range(tp_size):
            assert torch.equal(
                fragments._fuse(whole, widths, replicas, tp_size, rank),
                gathered[:, rank * per_rank : (rank + 1) * per_rank],
            )

    def test_a_fused_layer_nnsight_does_not_know_warns(self):
        # The module-level `importorskip("vllm")` answers with this very
        # directory when vLLM is not installed, so ask for the module the
        # dispatch actually needs.
        pytest.importorskip("vllm.model_executor.layers.linear")

        # The two fused layers vLLM ships are not a closed set — a QKV with an
        # indexer packs five — and one left in rank order has to say so.
        class Packed:
            output_partition_sizes = [64, 64]

        with pytest.warns(UserWarning, match="rank order"):
            assert fragments._fused_sub_shards(Packed()) is None
