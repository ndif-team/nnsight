"""Tensor parallelism: sharded activations read and edit as whole tensors.

Under TP a linear layer's value at any one rank is only that rank's shard. The
port gathers those shards before a worker sees the value and re-shards whatever
the worker leaves before vLLM's own forward carries on. These tests pin that down
by running the identical trace on an unsharded reference (``vllm_qwen_ref``, one
rank) and on the sharded engine (``vllm_qwen_tp``) and comparing.

Two sharding styles behave differently and are tested accordingly:

* Row-parallel layers (``o_proj``, ``down_proj``) split the contraction and
  all-reduce their output, so the whole a worker sees is laid out exactly as the
  single-rank run's — values compare element-wise and a positional edit is the
  same logical edit on both.
* Column-parallel layers here are *fused* (``qkv_proj`` packs Q/K/V,
  ``gate_up_proj`` packs gate/up), and each rank holds a slice of every packed
  part. Gathering concatenates the slices in rank order, so the whole holds all
  the same values as the single-rank run but grouped differently. It is checked
  as a per-row multiset rather than element-wise.

The whole module is skipped unless the machine has >=2 GPUs.
"""

import pytest
import torch

pytest.importorskip("vllm")

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
    def test_column_parallel_gathers_every_value(
        self, vllm_qwen_ref, vllm_qwen_tp, path, ET_prompt
    ):
        ref_hs, ref_logits = _read(vllm_qwen_ref, path, ET_prompt)
        tp_hs, tp_logits = _read(vllm_qwen_tp, path, ET_prompt)

        # Full width, and every value present — the fused packing means the
        # columns are grouped by rank, so compare the per-row sorted values.
        assert tp_hs.shape == ref_hs.shape
        assert tp_logits.argmax(dim=-1).item() == ref_logits.argmax(dim=-1).item()

        ref_sorted = ref_hs.sort(dim=-1).values
        tp_sorted = tp_hs.sort(dim=-1).values
        assert _min_row_cosine(tp_sorted, ref_sorted) > 0.99

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


class TestShardedParameters:
    """A parameter read inside a trace is the full tensor, not this rank's shard."""

    @pytest.mark.parametrize("path", ROW_PARALLEL)
    @torch.no_grad()
    def test_row_weight_matches_reference(
        self, vllm_qwen_ref, vllm_qwen_tp, path, ET_prompt
    ):
        with vllm_qwen_ref.trace(ET_prompt, temperature=0.0, top_p=1):
            ref = _submodule(vllm_qwen_ref, path).weight.data.save()
        with vllm_qwen_tp.trace(ET_prompt, temperature=0.0, top_p=1):
            tp = _submodule(vllm_qwen_tp, path).weight.data.save()

        # A row-parallel weight splits the input dim; gathered in rank order it
        # is laid out exactly as the single-rank weight.
        assert tp.shape == ref.shape
        assert torch.equal(tp.cpu(), ref.cpu())

    @pytest.mark.parametrize("path", COLUMN_PARALLEL)
    @torch.no_grad()
    def test_column_weight_gathers_every_row(
        self, vllm_qwen_ref, vllm_qwen_tp, path, ET_prompt
    ):
        with vllm_qwen_ref.trace(ET_prompt, temperature=0.0, top_p=1):
            ref = _submodule(vllm_qwen_ref, path).weight.data.save()
        with vllm_qwen_tp.trace(ET_prompt, temperature=0.0, top_p=1):
            tp = _submodule(vllm_qwen_tp, path).weight.data.save()

        # Fused column-parallel weights come back with rows grouped by rank, the
        # layout the gathered output has, so compare as a multiset of rows.
        assert tp.shape == ref.shape
        ref_rows = ref.float().cpu().sum(dim=-1).sort().values
        tp_rows = tp.float().cpu().sum(dim=-1).sort().values
        assert torch.allclose(tp_rows, ref_rows)

    @torch.no_grad()
    def test_lm_head_weight_full_vocab(self, vllm_qwen_ref, vllm_qwen_tp, ET_prompt):
        with vllm_qwen_ref.trace(ET_prompt, temperature=0.0, top_p=1):
            ref = vllm_qwen_ref.lm_head.weight.data.save()
        vocab = ref.shape[0]
        tid = vocab - 5  # a row the last rank owns

        with vllm_qwen_tp.trace(ET_prompt, temperature=0.0, top_p=1):
            tp = vllm_qwen_tp.lm_head.weight.data.save()
            tp_row = vllm_qwen_tp.lm_head.weight[tid].save()

        # Full vocab with vLLM's TP padding rows dropped, and the upper-shard
        # row is the real token's row, not an index into this rank's slice.
        assert tp.shape == ref.shape
        assert torch.equal(tp_row.cpu(), ref[tid].cpu())

    @torch.no_grad()
    def test_weight_outside_trace_is_slice(self, vllm_qwen_ref, vllm_qwen_tp):
        # The gather is a trace-time correction; outside one the module holds
        # this rank's slice, as anywhere else in vLLM.
        tp = vllm_qwen_tp.model.layers[LAYER].mlp.down_proj.weight
        ref = vllm_qwen_ref.model.layers[LAYER].mlp.down_proj.weight
        ranks = vllm_qwen_tp.model.layers[LAYER].mlp.down_proj._module.tp_size
        assert tp.shape[1] * ranks == ref.shape[1]
