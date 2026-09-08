"""Ownership derivation and resolution (pp.py), single process.

Ownership comes from where modules are real, never from what they are named:
``derive_owners`` reduces the allgathered per-stage real-module lists with the
exactly-one-stage rule, and ``PPModuleMap`` resolves envoy-path strings
against the result. The module trees here use unconventional names
(``decoder_blocks``, ``word_embeddings``, ``output_projection``) so any
naming-convention dependence in the resolution would fail these tests.
"""

import pytest
import torch.nn as nn

from nnsight.modeling.vllm.pp import (
    PPModuleMap,
    derive_owners,
    is_pp_missing,
    resolve_meta,
    stub_rank_gated_modules,
)


class TestDeriveOwners:
    def test_module_real_on_one_stage_is_owned_by_it(self):
        owners = derive_owners(
            [
                {"decoder_blocks.0": {}, "word_embeddings": {}},
                {"decoder_blocks.1": {}, "output_projection": {}},
            ]
        )
        assert owners == {
            "decoder_blocks.0": 0,
            "word_embeddings": 0,
            "decoder_blocks.1": 1,
            "output_projection": 1,
        }

    def test_module_real_on_several_stages_is_dropped(self):
        # Containers and build-on-every-rank modules cannot be attributed;
        # they must resolve to None (treated as local), never to stage 0.
        owners = derive_owners(
            [
                {"": {}, "decoder_blocks": {}, "decoder_blocks.0": {}},
                {"": {}, "decoder_blocks": {}, "decoder_blocks.1": {}},
            ]
        )
        assert owners == {"decoder_blocks.0": 0, "decoder_blocks.1": 1}

    def test_three_stages(self):
        owners = derive_owners(
            [{"decoder_blocks.0": {}}, {"decoder_blocks.1": {}}, {"decoder_blocks.2": {}}]
        )
        assert owners["decoder_blocks.2"] == 2

    def test_empty_input(self):
        assert derive_owners([]) == {}
        assert derive_owners([{}, {}]) == {}


class TestPPModuleMap:
    @pytest.fixture
    def module_map(self):
        m = PPModuleMap(2)
        m.set_derived_owners(
            {
                "word_embeddings": 0,
                "decoder_blocks.0": 0,
                "decoder_blocks.3": 1,
                "final_norm": 1,
                "output_projection": 1,
            }
        )
        return m

    def test_resolves_envoy_paths(self, module_map):
        assert module_map.get_owning_rank("model.decoder_blocks.3.output") == 1
        assert module_map.get_owning_rank("model.word_embeddings.output") == 0

    def test_strips_each_eproperty_suffix(self, module_map):
        for suffix in ("output", "input", "inputs"):
            assert (
                module_map.get_owning_rank(f"model.output_projection.{suffix}") == 1
            )

    def test_submodule_inherits_nearest_owned_ancestor(self, module_map):
        path = "model.decoder_blocks.3.self_attention.qkv_proj.output"
        assert module_map.get_owning_rank(path) == 1

    def test_raw_names_resolve_without_the_root_component(self, module_map):
        assert module_map.get_owning_rank("decoder_blocks.0.output") == 0

    def test_unknown_path_is_treated_as_local_on_every_rank(self, module_map):
        assert module_map.get_owning_rank("model.rotary_embedding.output") is None
        assert module_map.is_local("model.rotary_embedding.output", 0)
        assert module_map.is_local("model.rotary_embedding.output", 1)

    def test_is_local_matches_the_owner(self, module_map):
        assert module_map.is_local("model.decoder_blocks.0.output", 0)
        assert not module_map.is_local("model.decoder_blocks.0.output", 1)

    def test_empty_map_resolves_everything_as_local(self):
        m = PPModuleMap(2)
        assert m.get_owning_rank("model.decoder_blocks.3.output") is None
        assert m.is_local("model.decoder_blocks.3.output", 0)

    def test_custom_root_path(self):
        m = PPModuleMap(2, root_path="engine")
        m.set_derived_owners({"decoder_blocks.3": 1})
        assert m.get_owning_rank("engine.decoder_blocks.3.output") == 1

    def test_installing_new_owners_replaces_memoized_results(self):
        m = PPModuleMap(2)
        m.set_derived_owners({"decoder_blocks.3": 1})
        assert m.get_owning_rank("model.decoder_blocks.3.output") == 1
        m.set_derived_owners({"decoder_blocks.3": 0})
        assert m.get_owning_rank("model.decoder_blocks.3.output") == 0


class TestIsPPMissing:
    def test_detects_the_stub_by_class_name(self):
        # Checked by name so nnsight needs no import of the vLLM internal
        # class; a stand-in with the same name must count.
        PPMissingLayer = type("PPMissingLayer", (nn.Identity,), {})
        assert is_pp_missing(PPMissingLayer())

    def test_real_modules_are_not_stubs(self):
        assert not is_pp_missing(nn.Identity())
        assert not is_pp_missing(nn.Linear(2, 2))


class TestResolveMeta:
    META = {"decoder_blocks.3": {"dtype": "bf16"}}

    def test_exact_raw_name(self):
        assert resolve_meta(self.META, "decoder_blocks.3") == {"dtype": "bf16"}

    def test_envoy_path_strips_the_root_once(self):
        assert resolve_meta(self.META, "model.decoder_blocks.3") == {"dtype": "bf16"}

    def test_never_strips_more_than_the_root(self):
        # Stripping arbitrary components could hit a wrong entry and stamp a
        # wrong dtype; a deeper miss must stay a miss.
        assert resolve_meta(self.META, "wrapper.model.decoder_blocks.3") is None

    def test_custom_root(self):
        assert (
            resolve_meta(self.META, "engine.decoder_blocks.3", root="engine")
            == {"dtype": "bf16"}
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))


# ---------------------------------------------------------------------------
# Rank-gated modules stubbed from the meta tree
# ---------------------------------------------------------------------------


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
