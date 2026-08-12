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
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dispatch=True,
    )
    yield model

    del model
    gc.collect()
    torch.cuda.empty_cache()


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
@torch.no_grad()
def test_experts_output_read_is_the_full_value(vllm_moe_tp):
    # The block output IS the all-reduce of the per-rank partials, so the two
    # halves of experts.output sum to it exactly when the read was gathered.
    with vllm_moe_tp.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
        mlp = vllm_moe_tp.model.layers[LAYER].mlp
        experts_out = mlp.experts.output.save()
        block_out = _block_output(mlp)

    experts_sum = (experts_out[0] + experts_out[1]).float().cpu()
    block = block_out.float().cpu()

    cosine = torch.nn.functional.cosine_similarity(
        experts_sum.flatten(), block.flatten(), dim=0
    ).item()
    max_delta = (experts_sum - block).abs().max().item()

    assert cosine >= READ_MIN_COSINE, (
        "experts.output read back a per-rank partial rather than the full "
        f"value: cos={cosine:.6f} against its own block output"
    )
    assert max_delta <= READ_MAX_DELTA, (
        f"experts.output diverges from its own block output: "
        f"max|delta|={max_delta:.4f}"
    )


@requires_two_gpus
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
        from vllm.model_executor.layers.fused_moe import FusedMoE

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
