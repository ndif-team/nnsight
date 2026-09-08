"""Stubbing modules the meta tree has and a rank's local tree does not."""

import torch.nn as nn

from nnsight.modeling.vllm.pp import is_pp_missing, stub_rank_gated_modules


def _tree(with_head: bool, real_blocks: bool):
    from vllm.model_executor.models.utils import PPMissingLayer

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_blocks = nn.ModuleList(
                [nn.Sequential(nn.Linear(2, 2)) for _ in range(2)]
                if real_blocks
                else [PPMissingLayer(), PPMissingLayer()]
            )

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Inner()
            if with_head:
                self.output_projection = nn.Linear(2, 2)
                self.score_scaler = nn.Identity()

    return Root()


def test_missing_rank_gated_attributes_get_stubs():
    meta = _tree(with_head=True, real_blocks=True)
    local = _tree(with_head=False, real_blocks=True)

    stubbed = stub_rank_gated_modules(local, meta)

    assert set(stubbed) == {"output_projection", "score_scaler"}
    assert is_pp_missing(local.output_projection)
    assert is_pp_missing(local.score_scaler)
    # The stubs register as submodules, so path lookups resolve.
    assert "output_projection" in dict(local.named_modules())


def test_modules_under_a_stage_stub_are_left_to_the_graft():
    meta = _tree(with_head=True, real_blocks=True)
    local = _tree(with_head=True, real_blocks=False)

    # Meta has core.decoder_blocks.N.0 under each block; locally each block
    # is a stage stub, whose subtree the meta-envoy graft provides.
    stubbed = stub_rank_gated_modules(local, meta)

    assert stubbed == []
    assert is_pp_missing(local.core.decoder_blocks[0])


def test_matching_trees_change_nothing():
    meta = _tree(with_head=True, real_blocks=True)
    local = _tree(with_head=True, real_blocks=True)
    assert stub_rank_gated_modules(local, meta) == []


if __name__ == "__main__":
    test_missing_rank_gated_attributes_get_stubs()
    test_modules_under_a_stage_stub_are_left_to_the_graft()
    test_matching_trees_change_nothing()
    print("OK")
