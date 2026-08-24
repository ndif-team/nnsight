"""Mixture-of-experts: deferred-reduce partials gather on read and re-split on write.

Qwen-MoE-family models build their fused-experts module with
``reduce_results=False``: ``mlp.experts`` returns per-rank *partial* sums (this
rank's shared-expert and routed-experts contributions) and the outer block
all-reduces them a few lines later. That is the same exposure
``RowParallelLinear`` has always had handling for, so intervention code reading
or swapping ``mlp.experts.output`` must see and produce the whole value.

Without the batcher's ``FusedMoE`` handling a read ships one rank's partial —
right shape, wrong numbers, no error — and a swapped-in value is double-counted
by the downstream all-reduce.

Both of vLLM's expert layouts over the same ranks are covered, since they reach
the group size differently:

* **tensor-sliced experts** (``enable_expert_parallel=False``): every rank holds
  a slice of every expert's matrices, so ``tp_size`` is the group.
* **whole-expert placement** (``enable_expert_parallel=True``): each rank holds
  ``num_experts / world_size`` whole experts, so the module-internal ``tp_size``
  drops to 1 and ``ep_size`` becomes the group.

The end-to-end tests need >=2 GPUs with room for the checkpoint — the deferred
reduce only exists at ``tp_size * ep_size > 1``, so there is nothing to run on
one rank. `TestBatcherPolicy` pins the same decisions against a stub and runs
anywhere vLLM imports.
"""

import gc

import pytest
import torch

pytest.importorskip("vllm")

MODEL = "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "The Eiffel Tower is located in the city of"
LAYER = 11
# vLLM reserves this fraction of each card; the checkpoint is ~14 GiB in bf16
# split two ways, plus activations and KV cache.
GPU_MEMORY_UTILIZATION = 0.40
MIN_FREE_MIB = 36_000

# Kernel-noise scale for a bf16 activation compared against its own all-reduce.
# An ungathered per-rank partial lands around cos 0.6-0.93, far outside this.
READ_MIN_COSINE = 0.9999
READ_MAX_DELTA = 0.05
# The write check compares two constants (0.01 + 0.02), so only bf16 rounding
# and the all-reduce contribute. A missing write-back divide doubles it.
WRITE_MAX_DELTA = 0.005


def _gpus_with_free_memory(min_free_mib: int) -> int:
    """How many visible GPUs currently have ``min_free_mib`` free."""
    count = 0
    for index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(index)
        if free // (1024 * 1024) >= min_free_mib:
            count += 1
    return count


def _experts_output_is_partial() -> bool:
    """Whether this vLLM leaves a fused-experts output as a per-rank partial.

    Through 0.26 it does — the block all-reduces a few lines later, which is what
    the tests below use as their oracle. From 0.27 the layer reduces its own
    output (`MoERunner._maybe_all_reduce`), and measuring it on a two-rank
    Qwen1.5-MoE confirms it: both ranks hand back the identical tensor. There is
    then no partial to sum, no gather to check, and nothing to re-split — so
    these read something that does not exist there.

    What they check on 0.27 instead — that a read matches what the module really
    produced, and that a write reaches the block once when nothing is
    re-split — still wants writing.
    """
    from vllm.model_executor.layers import fused_moe

    return hasattr(fused_moe, "FusedMoE")


requires_partial_experts = pytest.mark.skipif(
    not _experts_output_is_partial(),
    reason=(
        "this vLLM reduces the fused-experts output itself, so there is no "
        "per-rank partial for these to assemble; 0.27-shaped equivalents are "
        "not written yet"
    ),
)

requires_two_gpus = pytest.mark.skipif(
    _gpus_with_free_memory(MIN_FREE_MIB) < 2,
    reason=f"needs 2 GPUs with {MIN_FREE_MIB} MiB free each",
)


@pytest.fixture(
    scope="module",
    params=[False, True],
    ids=["tensor_sliced_experts", "whole_expert_placement"],
)
def vllm_moe_tp(request):
    """A two-rank MoE engine in one of the two expert layouts.

    Torn down between layouts: two engines of this size do not fit alongside each
    other, and pytest builds the next parameter's engine before it would collect
    the previous one.
    """
    from nnsight.modeling.vllm import VLLM

    model = VLLM(
        MODEL,
        tensor_parallel_size=2,
        enable_expert_parallel=request.param,
        # Off so an installed block sees whole prompts (a cached token runs no
        # forward). The traced reads below are single-prompt and unaffected.
        enable_prefix_caching=False,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dispatch=True,
    )
    yield model

    del model
    gc.collect()
    torch.cuda.empty_cache()


def _assert_gathered(experts_out, block_out, what):
    """The two per-rank partials sum to the block's own all-reduced output.

    Which they only do if the read was gathered — an ungathered partial is the
    right shape and roughly half the value.
    """
    experts_sum = (experts_out[0] + experts_out[1]).float().cpu()
    block = block_out.float().cpu()

    cosine = torch.nn.functional.cosine_similarity(
        experts_sum.flatten(), block.flatten(), dim=0
    ).item()
    max_delta = (experts_sum - block).abs().max().item()

    assert cosine >= READ_MIN_COSINE, (
        f"{what} read back a per-rank partial rather than the full value: "
        f"cos={cosine:.6f} against its own block output"
    )
    assert max_delta <= READ_MAX_DELTA, (
        f"{what} diverges from its own block output: max|delta|={max_delta:.4f}"
    )


def _block_output(mlp):
    """The block's own output, cloned at access.

    vLLM's next layer mutates the returned hidden-states tensor in place (fused
    add_rms_norm), so a raw save reads back corrupted values. That save-time
    hazard is the separately tracked clone-on-save bug (#661); cloning here keeps
    the oracle measuring the batcher and nothing else. The experts tuple needs no
    clone — the gather's all-reduce already allocates fresh tensors.
    """
    return mlp.output.clone().save()


@requires_two_gpus
@requires_partial_experts
@torch.no_grad()
def test_experts_output_read_is_the_full_value(vllm_moe_tp):
    # The block output IS the all-reduce of the per-rank partials, so the two
    # halves of experts.output sum to it exactly when the read was gathered.
    with vllm_moe_tp.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
        mlp = vllm_moe_tp.model.layers[LAYER].mlp
        experts_out = mlp.experts.output.save()
        block_out = _block_output(mlp)

    _assert_gathered(experts_out, block_out, "experts.output")


@requires_two_gpus
@requires_partial_experts
@torch.no_grad()
def test_swapped_experts_output_reaches_the_block_once(vllm_moe_tp):
    # The block computes all_reduce(a' + b') from whatever the write-back left on
    # each rank. Correct re-splitting gives back a + b; skipping the divide
    # double-counts to 2(a + b).
    with vllm_moe_tp.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
        mlp = vllm_moe_tp.model.layers[LAYER].mlp
        shared = mlp.experts.output[0]
        a = torch.full_like(shared, 0.01)
        b = torch.full_like(shared, 0.02)
        mlp.experts.output = (a, b)
        swapped_block = _block_output(mlp)

    block = swapped_block.float().cpu()
    expected = torch.full_like(block, 0.03)
    max_delta = (block - expected).abs().max().item()

    assert max_delta <= WRITE_MAX_DELTA, (
        "a swapped experts.output did not reach the block exactly once "
        f"(a missing write-back divide double-counts it): "
        f"max|delta|={max_delta:.4f}"
    )


@requires_two_gpus
@requires_partial_experts
@torch.no_grad()
def test_each_invoke_reads_its_own_rows(vllm_moe_tp):
    # Continuous batching packs both requests into one flat slab; the gather runs
    # once over the whole of it and the rows are handed back per request. If the
    # split and the gather disagreed, one invoke would read the other's rows —
    # still summing to *a* block output, but not to its own.
    short, long = "Paris", PROMPT

    with vllm_moe_tp.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
        with tracer.invoke(short):
            short_mlp = vllm_moe_tp.model.layers[LAYER].mlp
            short_experts = short_mlp.experts.output.save()
            short_block = _block_output(short_mlp)
        with tracer.invoke(long):
            long_mlp = vllm_moe_tp.model.layers[LAYER].mlp
            long_experts = long_mlp.experts.output.save()
            long_block = _block_output(long_mlp)

    # Each request's own prompt length, not the batch's.
    assert short_experts[0].shape[0] < long_experts[0].shape[0]
    assert short_experts[0].shape[0] == short_block.shape[0]
    assert long_experts[0].shape[0] == long_block.shape[0]

    _assert_gathered(short_experts, short_block, "the short invoke's read")
    _assert_gathered(long_experts, long_block, "the long invoke's read")


@requires_two_gpus
@requires_partial_experts
@torch.no_grad()
def test_an_installed_block_reads_the_full_value(vllm_moe_tp):
    # An edit runs per request on every rank, the same as a trace's block, so its
    # read has to be gathered too — and it arrives on the request's output rather
    # than in a variable, from whichever ranks reported it.
    with vllm_moe_tp.edit() as (tracer, edit):
        mlp = vllm_moe_tp.model.layers[LAYER].mlp
        experts_out = mlp.experts.output.save()
        block_out = _block_output(mlp)
    try:
        saves = vllm_moe_tp.generate(
            [PROMPT], max_tokens=1, temperature=0.0, top_p=1
        )[0].saves

        _assert_gathered(
            saves["experts_out"], saves["block_out"], "an installed block's read"
        )
    finally:
        edit.clear()


class TestFragmentPolicy:
    """Which MoE values count as pieces, and what a write-back is divided by.

    Runs without GPUs: the decision in `_is_piece` and the divisor in
    `VLLMFragments.fragment` are pure functions of the module's parallel config,
    so a stub with that config pins them. The end-to-end tests above check the
    numbers these decisions produce; these check the decisions.
    """

    @staticmethod
    def _stub(tp, ep, reduce_results=False, must_reduce=False):
        """A real ``FusedMoE`` (so ``isinstance`` holds) with its config stubbed.

        ``FusedMoE.__init__`` builds quant methods and reads the live process
        group, and ``tp_size``/``ep_size`` are properties off that group — none of
        which exists off-engine, so bypass it and answer for them directly.
        """
        FusedMoE = getattr(
            __import__("vllm.model_executor.layers.fused_moe", fromlist=["x"]),
            "FusedMoE",
            None,
        )
        if FusedMoE is None:
            pytest.skip("this vLLM has no FusedMoE; see TestModularMoEPolicy")

        class StubMoE(FusedMoE):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.reduce_results = reduce_results

            @property
            def tp_size(self):
                return tp

            @property
            def ep_size(self):
                return ep

            def must_reduce_shared_expert_outputs(self):
                return must_reduce

        return StubMoE()

    @pytest.mark.parametrize(
        "config, kind, expected",
        [
            # A deferred combine leaves a partial, under either expert layout.
            (dict(tp=2, ep=1), "output", True),
            (dict(tp=1, ep=2), "output", True),
            # Hidden states and router logits are replicated on every rank.
            (dict(tp=2, ep=1), "input", False),
            # reduce_results=True (Mixtral) all-reduces inside forward.
            (dict(tp=2, ep=1, reduce_results=True), "output", False),
            # One rank: nothing was split, so nothing to gather.
            (dict(tp=1, ep=1), "output", False),
            # A combine kernel that already reduced leaves nothing to gather.
            (dict(tp=2, ep=1, must_reduce=True), "output", False),
        ],
    )
    def test_only_deferred_reduce_outputs_are_pieces(self, config, kind, expected):
        from nnsight.modeling.vllm.fragments import _is_piece

        assert _is_piece(self._stub(**config), kind) is expected

    @pytest.mark.parametrize("tp, ep", [(2, 1), (1, 2), (2, 4)])
    def test_write_back_divides_by_the_group_size(self, tp, ep):
        # The block all-reduces over tp*ep ranks right after, so an equal share
        # is what sums back to the whole exactly once.
        from nnsight.modeling.vllm.fragments import VLLMFragments

        fragments = VLLMFragments()
        fragments.rules["m.output"] = (self._stub(tp=tp, ep=ep), "output")
        sharded = fragments.fragment("m.output", torch.full((2, 4), 6.0))

        assert torch.allclose(sharded, torch.full((2, 4), 6.0 / (tp * ep)))


class TestReplicatedLinearsAreNotPieces:
    """A `disable_tp=True` linear is replicated, so nothing about it is a piece.

    vLLM lets a model opt one layer out of tensor parallelism — DeepSeek-V2's
    `fused_qkv_a_proj` does, and the branch beside it uses a `ReplicatedLinear`
    for the same role — by setting the layer's own ``tp_size`` to 1 while the
    engine stays sharded. Its own forward guards every collective on
    ``tp_size > 1`` for that reason. Reading the engine's world size instead
    would gather a tensor every rank already holds whole: the value comes back
    ``world_size``x too wide, built of identical copies, and an edit written
    against it goes into the next matmul at the wrong shape — silently, both ways.

    Runs without GPUs, like `TestFragmentPolicy`: the decision is a pure function
    of the module's parallel config. No cached checkpoint uses `disable_tp`, so a
    stub is the only way to pin it.
    """

    @staticmethod
    def _stub(cls, tp, **attributes):
        """A real linear subclass (so ``isinstance`` holds) with its config stubbed.

        The real ``__init__`` reads the live process group, which does not exist
        off-engine, so bypass it and set what `_is_piece` reads.
        """

        class Stub(cls):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.tp_size = tp
                for name, value in attributes.items():
                    setattr(self, name, value)

        return Stub()

    @pytest.mark.parametrize("side", ["input", "output"])
    def test_a_replicated_column_linear_is_never_a_piece(self, side):
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        from nnsight.modeling.vllm.fragments import _is_piece

        # gather_output=False is what makes a *sharded* column linear a piece.
        replicated = self._stub(ColumnParallelLinear, tp=1, gather_output=False)
        sharded = self._stub(ColumnParallelLinear, tp=2, gather_output=False)

        assert _is_piece(replicated, side) is False
        assert _is_piece(sharded, side) is (side == "output")

    @pytest.mark.parametrize("side", ["input", "output"])
    def test_a_replicated_row_linear_is_never_a_piece(self, side):
        from vllm.model_executor.layers.linear import RowParallelLinear

        from nnsight.modeling.vllm.fragments import _is_piece

        config = dict(input_is_parallel=True, reduce_results=False)
        replicated = self._stub(RowParallelLinear, tp=1, **config)
        sharded = self._stub(RowParallelLinear, tp=2, **config)

        assert _is_piece(replicated, side) is False
        assert _is_piece(sharded, side) is True


class TestModularMoEPolicy:
    """The 0.27 layer reduces its own output; what it leaves partial, and when.

    vLLM 0.27 rebuilt the fused-experts layer around a factory and a modular
    kernel. The flags that used to say whether the output was still a per-rank
    partial are gone, replaced by the guard in ``MoERunner._maybe_all_reduce`` —
    so the layer is whole unless one of that guard's conditions says otherwise.

    Runs on any vLLM: `_is_piece` picks its branch on whether the layer carries a
    ``moe_config``, so a stub with one exercises the 0.27 rule wherever it is run.
    """

    @staticmethod
    def _stub(tp=2, ep=1, skip_final_all_reduce=False, is_sequence_parallel=False,
              fused_output_is_reduced=False):
        """A layer with the config `_is_piece` reads, and nothing else.

        Deliberately not a subclass of the real one: 0.27's `MoERunner` is
        abstract and answers some of these names with read-only properties, and
        0.26's class does not exist on 0.27. `_is_piece` finds the class through
        `_moe_layer`, so the test points that at the stub instead — which leaves
        the policy itself as the only thing under test, on either version.
        """

        class Parallel:
            pass

        class Config:
            pass

        parallel = Parallel()
        parallel.tp_size, parallel.ep_size = tp, ep
        config = Config()
        config.moe_parallel_config = parallel
        config.skip_final_all_reduce = skip_final_all_reduce
        config.is_sequence_parallel = is_sequence_parallel

        class Stub:
            pass

        stub = Stub()
        stub.moe_config = config
        stub._fused_output_is_reduced = fused_output_is_reduced
        return stub

    @staticmethod
    def _is_piece(stub, side):
        from unittest import mock

        from nnsight.modeling.vllm import fragments

        with mock.patch.object(
            fragments, "_moe_layer", return_value=type(stub)
        ):
            return fragments._is_piece(stub, side)

    @pytest.mark.parametrize(
        "config, expected, why",
        [
            (dict(), False, "the layer all-reduces its own output"),
            (dict(skip_final_all_reduce=True), True, "the reduce was deferred"),
            (dict(skip_final_all_reduce=True, tp=1, ep=1), False, "one rank"),
            (dict(skip_final_all_reduce=True, fused_output_is_reduced=True), False,
             "the kernel already reduced it"),
            (dict(skip_final_all_reduce=True, is_sequence_parallel=True), False,
             "split by rows, not a partial sum"),
        ],
    )
    def test_only_a_deferred_reduce_leaves_a_piece(self, config, expected, why):
        assert self._is_piece(self._stub(**config), "output") is expected, why

    def test_the_input_is_never_a_piece(self):
        assert self._is_piece(self._stub(skip_final_all_reduce=True), "input") is False

    def test_the_group_size_comes_off_the_config(self):
        from nnsight.modeling.vllm.fragments import _moe_group_size

        # Expert parallelism moves the ranks from tp to ep; the product is the
        # group either way.
        assert _moe_group_size(self._stub(tp=2, ep=1)) == 2
        assert _moe_group_size(self._stub(tp=1, ep=4)) == 4


class TestDCPGroupGather:
    """A column layer sharded across DCP groups: the gather drops the replicas."""

    def test_one_shard_per_group_survives(self):
        from nnsight.modeling.vllm.fragments import _one_per_group

        a, b = torch.full((3, 4), 1.0), torch.full((3, 4), 2.0)
        gathered = torch.cat([a, a, b, b], dim=-1)  # world 4, groups of 2
        whole = _one_per_group(gathered, world=4, group_size=2)
        assert torch.equal(whole, torch.cat([a, b], dim=-1))

    def test_a_group_of_one_is_the_plain_gather(self):
        from nnsight.modeling.vllm.fragments import _one_per_group

        gathered = torch.arange(8.0).reshape(1, 8)
        assert torch.equal(_one_per_group(gathered, world=4, group_size=1), gathered)
