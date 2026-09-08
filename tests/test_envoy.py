
import pytest
import torch
import nnsight
import torch.nn as nn

from nnsight import NNsightDeprecationWarning

from nnsight.intervention.envoy import Envoy, traceable


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.act = nn.ReLU()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.mlp = MLP()


class Model(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.embed = nn.Linear(8, 8)
        self.layers = nn.ModuleList([Block() for _ in range(n_layers)])
        self.head = nn.Linear(8, 8)


def module_paths(module):
    return {"model" if name == "" else f"model.{name}" for name, _ in module.named_modules()}


@pytest.fixture
def module():
    return Model()


@pytest.fixture
def envoy(module):
    return Envoy(module)


class TestOutsideInterleaving:
    def test_output_outside_a_trace(self, envoy):
        # Reading an activation with no run to read it from names the location and
        # the reason. Not an AttributeError: raised from a property, that is taken
        # for "no such attribute" and comes back out of __getattr__ naming `output`.
        with pytest.raises(ValueError, match="outside of interleaving"):
            envoy.embed.output

    def test_input_outside_a_trace(self, envoy):
        with pytest.raises(ValueError, match="outside of interleaving"):
            envoy.embed.input


class TestTree:
    def test_root_path(self, envoy):
        assert envoy.path == "model"

    def test_mirrors_module_tree(self, envoy, module):
        paths = {node.path for node in envoy.modules()}
        assert paths == module_paths(module)

    @pytest.mark.parametrize(
        "expected",
        [
            "model.embed",
            "model.layers",
            "model.layers.0",
            "model.layers.0.mlp",
            "model.layers.0.mlp.fc",
            "model.head",
        ],
    )
    def test_known_paths_present(self, envoy, expected):
        paths = {node.path for node in envoy.modules()}
        assert expected in paths

    def test_attribute_access(self, envoy):
        assert isinstance(envoy.embed, Envoy)
        assert isinstance(envoy.layers, Envoy)
        assert envoy.embed.path == "model.embed"

    def test_modulelist_children_indexed(self, envoy):
        layer0 = getattr(envoy.layers, "0")
        assert isinstance(layer0, Envoy)
        assert layer0.path == "model.layers.0"
        assert layer0.mlp.fc.path == "model.layers.0.mlp.fc"

    def test_iter_yields_direct_children(self, envoy, module):
        # __iter__ is direct children only, not the whole subtree.
        assert [e._module for e in envoy] == [m for _, m in module.named_children()]

    def test_iter_modulelist_yields_layers(self, envoy, module):
        # `for layer in model.layers:` yields each block, not the ModuleList
        # itself or nested submodules.
        layers = list(envoy.layers)
        assert [layer._module for layer in layers] == list(module.layers)
        assert [layer.path for layer in layers] == [
            f"model.layers.{i}" for i in range(len(module.layers))
        ]

    def test_modules_matches_module_count(self, envoy, module):
        # modules() is the recursive walk over the whole tree.
        assert len(envoy.modules()) == len(list(module.named_modules()))

    def test_envoy_wraps_correct_module(self, envoy, module):
        assert envoy.embed._module is module.embed
        assert getattr(envoy.layers, "0")._module is module.layers[0]


class WithExtras(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.config = "cfg"

    def forward(self, x):
        return self.fc(x)

    def describe(self):
        return "hello"


class TestAttributePassthrough:
    def test_non_module_attr(self):
        envoy = Envoy(WithExtras())
        assert envoy.config == "cfg"

    def test_method_passthrough(self):
        envoy = Envoy(WithExtras())
        assert envoy.describe() == "hello"

    def test_child_resolves_without_passthrough(self):
        envoy = Envoy(WithExtras())
        assert isinstance(envoy.fc, Envoy)

    def test_missing_attr_raises(self):
        envoy = Envoy(WithExtras())
        with pytest.raises(AttributeError):
            envoy.does_not_exist

    def test_unregistered_module_is_wrapped(self):
        model = WithExtras()
        envoy = Envoy(model)
        # a module reachable on the torch module but not mirrored as a child
        object.__setattr__(model, "extra", nn.Linear(4, 4))
        wrapped = envoy.extra
        assert isinstance(wrapped, Envoy)
        assert wrapped.path == "model.extra"
        assert wrapped._module is model.extra


class TestRepr:
    def test_matches_torch_repr(self, envoy, module):
        assert repr(envoy) == repr(module)

    def test_modulelist_compression(self, envoy):
        assert "(0-2): 3 x Block(" in repr(envoy)

    def test_leaf_repr(self, envoy):
        assert repr(envoy.embed) == "Linear(in_features=8, out_features=8, bias=True)"

    def test_repr_tracks_added_module(self, envoy, module):
        envoy.extra = nn.Linear(4, 2)
        assert repr(envoy) == repr(module)
        assert "(extra): Linear(in_features=4, out_features=2, bias=True)" in repr(envoy)

    def test_repr_tracks_replaced_module(self, envoy, module):
        envoy.head = nn.Linear(8, 16)
        assert repr(envoy) == repr(module)


class TestSetattr:
    def test_added_module_becomes_envoy(self, envoy):
        envoy.extra = nn.Linear(8, 8)
        assert isinstance(envoy.extra, Envoy)
        assert envoy.extra.path == "model.extra"

    def test_added_module_registered_on_torch_module(self, envoy, module):
        layer = nn.Linear(8, 8)
        envoy.extra = layer
        assert module.extra is layer
        assert "model.extra" in {node.path for node in envoy.modules()}

    def test_added_nested_module_builds_subtree(self, envoy):
        envoy.extra = Block()
        paths = {node.path for node in envoy.modules()}
        assert "model.extra.mlp.fc" in paths

    def test_non_module_assignment_passthrough(self, envoy):
        envoy.label = "tag"
        assert envoy.label == "tag"

    def test_replace_dedupes_children(self, envoy, module):
        before = len(list(envoy))
        new_head = nn.Linear(8, 8)
        envoy.head = new_head
        assert len(list(envoy)) == before
        assert envoy.head._module is new_head
        assert module.head is new_head

    def test_replace_no_duplicate_paths(self, envoy):
        envoy.head = nn.Linear(8, 8)
        paths = [node.path for node in envoy.modules()]
        assert len(paths) == len(set(paths))


class Stack(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SharedStack(nn.Module):
    """A ModuleList holding one module twice — torch still indexes three entries."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)
        self.layers = nn.ModuleList([self.shared, nn.Linear(8, 8), self.shared])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestSharedEntries:
    @pytest.fixture
    def shared(self):
        return Envoy(SharedStack())

    def test_every_entry_is_indexable(self, shared):
        # `named_children()` deduplicates by identity, so entry 2 used to be
        # missing entirely and `layers[2]` raised IndexError.
        assert len(list(shared.layers)) == 3
        assert shared.layers[1]._module is shared._module.layers[1]
        assert shared.layers[2]._module is shared._module.shared

    def test_a_shared_entry_is_the_one_envoy(self, shared):
        assert shared.layers[0] is shared.shared
        assert shared.layers[2] is shared.shared

    def test_a_shared_module_is_listed_once(self, shared):
        # The same rule torch's named_modules() follows.
        assert len(shared.modules()) == len(list(shared._module.named_modules()))

    def test_the_repr_shows_every_entry(self, shared):
        assert "(0-2): 3 x Linear" in repr(shared.layers)

    def test_a_shared_list_traces(self, shared):
        x = torch.randn(1, 8)
        with shared.trace(x):
            middle = shared.layers[1].output.save()
        module = shared._module
        assert torch.allclose(middle, module.layers[1](module.shared(x)))


class TestRebuiltContainer:
    """A container rebuilt from modules the tree already wraps — truncating layers."""

    @pytest.fixture
    def stack(self):
        return Envoy(Stack())

    def test_the_entries_are_kept(self, stack):
        stack.layers = nn.ModuleList(list(stack._module.layers)[:2])
        assert len(list(stack.layers)) == 2
        assert [layer.path for layer in stack.layers] == [
            "model.layers.0",
            "model.layers.1",
        ]

    def test_the_tree_still_mirrors_the_module(self, stack):
        stack.layers = nn.ModuleList(list(stack._module.layers)[:2])
        assert {node.path for node in stack.modules()} == module_paths(stack._module)

    def test_a_truncated_stack_traces(self, stack):
        stack.layers = nn.ModuleList(list(stack._module.layers)[:2])
        x = torch.randn(1, 8)
        with stack.trace(x):
            last = stack.layers[1].output.save()
        assert torch.allclose(last, stack._module(x))

    def test_gpt2_truncated_blocks(self):
        # Keeping the first four blocks of a real model is the ordinary way a
        # user hits this; every block is already wrapped, so the new ModuleList
        # used to end up with no children at all.
        from nnsight.modeling.transformers import TransformersModel

        model = TransformersModel(
            "openai-community/gpt2", task="text-generation", dispatch=True
        )
        model.transformer.h = nn.ModuleList(list(model.transformer._module.h)[:4])
        assert len(list(model.transformer.h)) == 4
        with model.trace("Hello"):
            hidden = model.transformer.h[3].output.save()
        assert hidden.shape[-1] == model._module.config.n_embd


class TestReplacement:
    def test_a_replacement_keeps_its_index(self):
        # Remove-then-append put the new child last, shifting every index after
        # it, so `envoy[2]` named the module the wrapped module holds at 3.
        net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.Tanh())
        sequential = Envoy(net)
        setattr(sequential, "1", nn.Identity())  # `sequential[1] = ...` is not a thing
        assert [type(child._module) for child in sequential] == [type(m) for m in net]
        assert isinstance(sequential[1]._module, nn.Identity)
        assert isinstance(sequential[3]._module, nn.Tanh)

    def test_a_replaced_module_is_deregistered(self, envoy, module):
        # Left registered, the replaced module would come back as an alias of
        # its replacement and serve the replacement's values.
        old_module = module.head
        old_envoy = envoy.head  # held, so the registry entry can't just be collected
        envoy.head = nn.Identity()
        assert id(old_module) not in envoy.interleaver.envoys
        envoy.spare = old_module
        assert envoy.spare is not old_envoy
        assert envoy.spare.path == "model.spare"


class SelfNaming(nn.Module):
    """A property returning the module itself, as `base_model` does on a base model."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8)

    @property
    def base_model(self):
        return self

    def forward(self, x):
        return self.layer(x)


class TestSelfNamingAttribute:
    def test_it_resolves_to_this_envoy(self):
        envoy = Envoy(SelfNaming())
        assert envoy.base_model is envoy
        assert envoy.interleaver.envoys[id(envoy._module)] is envoy

    def test_it_adds_no_path_to_the_tree(self):
        envoy = Envoy(SelfNaming())
        before = {node.path for node in envoy.modules()}
        envoy.base_model
        assert {node.path for node in envoy.modules()} == before

    def test_a_rename_through_it_resolves(self):
        # A duplicate envoy re-ran `_bind_aliases` on the same module: RecursionError.
        envoy = Envoy(SelfNaming(), rename={"base_model.layer": "inner"})
        assert envoy.inner is envoy.layer


class TestTraceable:
    def _model(self):
        ran = []

        class Decorated(Envoy):
            @traceable
            def run(self, value):
                ran.append(value)
                return value

        return Decorated(nn.Linear(8, 8)), ran

    def test_direct_call_runs(self):
        model, ran = self._model()
        out = model.run(5)  # not a with block -> just runs
        assert out == 5
        assert ran == [5]

    def test_with_block_traces(self):
        model, ran = self._model()
        escaped = None
        with model.run(7):
            escaped = nnsight.save("body")
        assert ran == [7]  # the method ran via interleave
        assert escaped == "body"  # the block body ran too

    def test_called_while_interleaving_runs(self):
        model, ran = self._model()
        # Calling a traceable method from inside a trace just runs it.
        with model.trace(torch.randn(1, 8)):
            model.run(9)
        assert 9 in ran


class TestLookup:
    def test_getitem_indexes_children(self, envoy):
        assert envoy.layers[0].path == "model.layers.0"
        assert envoy.layers[2].path == "model.layers.2"

    def test_len_of_modulelist(self, envoy):
        assert len(envoy.layers) == 3

    def test_get_dotted_path_to_envoy(self, envoy):
        assert envoy.get("layers.0.mlp").path == "model.layers.0.mlp"

    def test_get_reaches_parameter(self, envoy):
        assert envoy.get("head.weight").shape == (8, 8)

    def test_modules_children_first_then_self(self, envoy):
        result = envoy.modules()
        assert result[-1] is envoy  # self comes last
        assert all(isinstance(node, Envoy) for node in result)

    def test_modules_matches_full_tree(self, envoy, module):
        assert {node.path for node in envoy.modules()} == module_paths(module)

    def test_modules_include_fn_filters(self, envoy):
        linears = envoy.modules(include_fn=lambda e: isinstance(e._module, nn.Linear))
        assert linears
        assert all(isinstance(e._module, nn.Linear) for e in linears)

    def test_named_modules_pairs_path_and_envoy(self, envoy, module):
        named = dict(envoy.named_modules())
        assert set(named) == module_paths(module)
        assert named["model.head"].path == "model.head"


class TestRename:
    def test_name_alias_binds_everywhere(self, module):
        # A single-component alias binds wherever that name resolves.
        envoy = Envoy(module, rename={"mlp": "block_mlp"})
        for i in range(len(envoy.layers)):
            assert envoy.layers[i].block_mlp is envoy.layers[i].mlp

    def test_original_name_still_works(self, module):
        envoy = Envoy(module, rename={"mlp": "block_mlp"})
        assert envoy.layers[0].mlp is envoy.layers[0].block_mlp

    def test_deep_path_mounts_on_root(self, module):
        envoy = Envoy(module, rename={"layers.0.mlp": "first_mlp"})
        assert envoy.first_mlp is envoy.layers[0].mlp
        # ...and only on the root (not on a block, which can't resolve the path).
        assert not hasattr(envoy.layers[1], "first_mlp")

    def test_modulelist_mount(self, module):
        envoy = Envoy(module, rename={"layers": "blocks"})
        assert envoy.blocks[0] is envoy.layers[0]

    def test_relative_path_alias(self, module):
        envoy = Envoy(module, rename={"layers.1": "second"})
        assert envoy.second is envoy.layers[1]

    def test_numeric_index_alias(self, module):
        envoy = Envoy(module, rename={"0": "zero"})
        assert envoy.layers.zero is envoy.layers[0]

    def test_multiple_aliases(self, module):
        envoy = Envoy(module, rename={"embed": ["e", "emb"]})
        assert envoy.e is envoy.embed
        assert envoy.emb is envoy.embed

    def test_leading_dot_is_noop(self, module):
        envoy = Envoy(module, rename={".mlp": "m"})
        assert envoy.layers[0].m is envoy.layers[0].mlp

    def test_aliases_do_not_double_count_tree(self, module):
        # Aliases are extra references, not new children.
        plain = {node.path for node in Envoy(module).modules()}
        aliased = {node.path for node in Envoy(module, rename={"mlp": "block_mlp"}).modules()}
        assert plain == aliased

    def test_repr_shows_direct_and_mounted_aliases(self, module):
        envoy = Envoy(module, rename={"embed": ["e", "emb"], "layers.0.mlp": "first_mlp"})
        r = repr(envoy)
        assert "e/emb/embed" in r  # direct child, decorated
        assert "(first_mlp):" in r  # deep mount, own line

    def test_get_resolves_alias(self, module):
        envoy = Envoy(module, rename={"layers": "blocks"})
        assert envoy.get("blocks.0.mlp") is envoy.layers[0].mlp

    # -- collisions ------------------------------------------------------
    #
    # Aliases bind with `object.__setattr__`, so before these checks anything
    # already under the name was overwritten in silence.

    def test_alias_shadowing_envoy_state_raises(self, module):
        # `path`, `interleaver` and the rest are set in `__init__`, so they are
        # instance attributes rather than class ones — the class-namespace scan
        # never saw them, and an alias overwrote them in silence.
        for state in ("path", "interleaver", "_module"):
            with pytest.raises(ValueError, match="would shadow"):
                Envoy(module, rename={"head": state})

    def test_alias_shadowing_a_module_attribute_raises(self, module):
        # An alias lands in the envoy's `__dict__`, which wins over the
        # `__getattr__` fallthrough — so it hides the wrapped model's own name
        # with nothing said. `eval` is `nn.Module`'s; `training` is its state.
        for owned in ("eval", "training", "parameters"):
            with pytest.raises(ValueError, match="would shadow"):
                Envoy(module, rename={"head": owned})

    def test_a_name_the_module_does_not_have_is_fine(self, module):
        # The check is the module's own surface, not a blocklist: a name it
        # never answered to is still a usable alias.
        envoy = Envoy(module, rename={"head": "readout"})
        assert envoy.readout.path == "model.head"

    def test_alias_shadowing_a_sibling_child_raises(self, module):
        # `head` and `embed` both exist; aliasing one to the other's name would
        # leave the original in the tree but unreachable by name, so an
        # intervention written against it would land on the wrong module.
        with pytest.raises(ValueError) as info:
            Envoy(module, rename={"head": "embed"})

        message = str(info.value)
        assert "rename" in message
        assert "'embed'" in message
        assert "child module" in message

    def test_alias_shadowing_an_envoy_attribute_raises(self, module):
        # Shadowing `trace` used to break `model.trace(...)` itself, surfacing
        # much later as a TypeError about the context manager protocol.
        with pytest.raises(ValueError) as info:
            Envoy(module, rename={"head": "trace"})

        assert "Envoy.trace" in str(info.value)

    @pytest.mark.parametrize("alias", ["output", "input", "inputs", "source"])
    def test_alias_shadowing_an_eproperty_raises(self, module, alias):
        # These are data descriptors whose __get__ raises outside interleaving,
        # so the conflict has to be found without reading them. Previously
        # `output` failed with "Cannot access `model.output` outside of
        # interleaving", which never mentions `rename`.
        with pytest.raises(ValueError) as info:
            Envoy(module, rename={"head": alias})

        assert "rename" in str(info.value)

    def test_two_paths_claiming_one_alias_raises(self, module):
        # Self-contradictory: one silently won by dict insertion order.
        with pytest.raises(ValueError) as info:
            Envoy(module, rename={"embed": "dup", "head": "dup"})

        assert "already bound" in str(info.value)

    # -- collisions that are not collisions ------------------------------

    def test_alias_equal_to_its_own_name_is_a_noop(self, module):
        # One `rename` is meant to be reusable across architectures that spell
        # the same module differently. Pairing `{"attn": "attn"}` with an alias
        # for the other spelling is the normal way to write that, so a key
        # aliased to its own name must stay harmless.
        envoy = Envoy(module, rename={"attn": "attn", "self_attn": "attn"})

        assert envoy.layers[0].attn is envoy.layers[0]._module.attn or isinstance(
            envoy.layers[0].attn, Envoy
        )

    def test_unresolvable_key_is_still_skipped(self, module):
        # The cross-architecture case: a key that names nothing here is skipped,
        # not an error.
        envoy = Envoy(module, rename={"layers": "blocks", "decoder.layers": "blocks"})

        assert envoy.blocks[0] is envoy.layers[0]

    def test_two_paths_to_the_same_module_share_an_alias(self, module):
        # Tied weights: two keys reaching one module bind the same object, so
        # the second is a no-op rather than a conflict.
        module.tied = module.head
        envoy = Envoy(module, rename={"head": "out", "tied": "out"})

        assert envoy.out is envoy.head


class TestDevice:
    def test_device_of_parameters(self, envoy):
        assert envoy.device == torch.device("cpu")

    def test_devices_set(self, envoy):
        assert envoy.devices == {torch.device("cpu")}

    def test_device_none_when_parameterless(self):
        assert Envoy(nn.ReLU()).device is None

    def test_devices_empty_when_parameterless(self):
        assert Envoy(nn.ReLU()).devices == set()

    def test_to_returns_self_for_chaining(self, envoy):
        assert envoy.to("cpu") is envoy

    def test_cpu_returns_self(self, envoy):
        assert envoy.cpu() is envoy


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        return self.b(self.a(x))


class TestResult:
    def test_result_is_the_forward_output(self):
        model = Envoy(nn.Linear(8, 8))
        x = torch.randn(2, 8)
        expected = model._module(x)
        with model.trace(x) as tracer:
            out = nnsight.save(tracer.result)
        assert torch.equal(out, expected)

    def test_result_after_a_module_output(self):
        model = Envoy(TwoLayer())
        x = torch.randn(2, 8)
        expected = model._module(x)
        with model.trace(x) as tracer:
            _ = model.a.output  # advance the run partway first
            out = nnsight.save(tracer.result)
        assert torch.equal(out, expected)

    def test_result_before_earlier_location_is_out_of_order(self):
        from nnsight.intervention.interleaver import OutOfOrderError

        model = Envoy(TwoLayer())
        x = torch.randn(2, 8)
        # Asking for the result first parks the worker on "result" for the whole
        # run, so a later request for an already-passed location is out of order.
        with pytest.raises(OutOfOrderError):
            with model.trace(x) as tracer:
                _ = tracer.result
                _ = model.a.output


class TestTracerCls:
    def test_trace_uses_custom_tracer_class(self):
        from nnsight.intervention.tracer import InterleavingTracer

        seen = []

        class MyTracer(InterleavingTracer):
            def execute(self, code):
                seen.append("trace")
                super().execute(code)

        envoy = Envoy(nn.Linear(8, 8))
        with envoy.trace(torch.randn(1, 8), tracer_cls=MyTracer) as tracer:
            out = envoy.output.save()
        assert isinstance(tracer, MyTracer)
        assert seen == ["trace"]
        assert out.shape[-1] == 8

    def test_session_uses_custom_tracer_class(self):
        from nnsight.tracing.tracer import Tracer

        seen = []

        class MySession(Tracer):
            def execute(self, code):
                seen.append("session")
                super().execute(code)

        envoy = Envoy(nn.Linear(8, 8))
        with envoy.session(tracer_cls=MySession) as session:
            _ = 1 + 1
        assert isinstance(session, MySession)
        assert seen == ["session"]


class TestDeprecatedIterAliases:
    def test_model_iter_warns_and_returns_iterations(self, envoy):
        from nnsight.intervention.iterator import Iterations

        with pytest.warns(NNsightDeprecationWarning, match="model.iter"):
            iterations = envoy.iter
        assert isinstance(iterations, Iterations)

    def test_model_all_warns_and_returns_iterations(self, envoy):
        from nnsight.intervention.iterator import Iterations

        with pytest.warns(NNsightDeprecationWarning, match=r"model\.all\(\)"):
            iterations = envoy.all()
        assert isinstance(iterations, Iterations)

    def test_all_warns_once(self, envoy):
        # all() builds the range directly, so only model.all() warns (not model.iter).
        with pytest.warns(NNsightDeprecationWarning) as record:
            envoy.all()
        assert sum("deprecated" in str(w.message) for w in record) == 1


class TestTraceBypass:
    def test_trace_false_returns_output(self):
        # trace=False bypasses tracing: run the module and return its output.
        envoy = Envoy(nn.Linear(8, 4))
        out = envoy.trace(torch.randn(2, 8), trace=False)
        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (2, 4)

    def test_trace_true_returns_tracer(self):
        from nnsight.intervention.tracer import InterleavingTracer

        envoy = Envoy(nn.Linear(8, 4))
        assert isinstance(envoy.trace(torch.randn(2, 8)), InterleavingTracer)


class TestMultipleWrappers:
    """Several independent Envoys (each its own Interleaver) over the SAME module."""

    def test_two_wrappers_agree(self):
        model = TwoLayer()
        w1, w2 = Envoy(model), Envoy(model)
        x = torch.randn(2, 8)
        with w1.trace(x) as t1:
            o1 = nnsight.save(t1.result)
        with w2.trace(x) as t2:
            o2 = nnsight.save(t2.result)
        assert torch.allclose(o1, o2)

    def test_modifications_are_independent(self):
        model = TwoLayer()
        w1, w2 = Envoy(model), Envoy(model)
        x = torch.randn(2, 8)
        with w1.trace(x) as t:
            baseline = nnsight.save(t.result)
        with w1.trace(x) as t:
            w1.a.output[:] = 0
            edited = nnsight.save(t.result)
        with w2.trace(x) as t:
            clean = nnsight.save(t.result)
        assert not torch.allclose(edited, baseline)
        assert torch.allclose(clean, baseline)

    def test_three_wrappers_agree(self):
        model = TwoLayer()
        x = torch.randn(2, 8)
        outs = []
        for _ in range(3):
            w = Envoy(model)
            with w.trace(x) as t:
                outs.append(nnsight.save(t.result))
        assert torch.allclose(outs[0], outs[1]) and torch.allclose(outs[1], outs[2])

    def test_intermediate_saves_match_across_wrappers(self):
        model = TwoLayer()
        w1, w2 = Envoy(model), Envoy(model)
        x = torch.randn(2, 8)
        with w1.trace(x):
            a1 = w1.a.output.save()
        with w2.trace(x):
            a2 = w2.a.output.save()
        assert torch.allclose(a1, a2)


class TranscoderSet(nn.Module):
    """A container that keeps its modules one level down, behind its own
    ``__getitem__`` — the shape a set of transcoders or SAEs takes."""

    def __init__(self, n: int = 4):
        super().__init__()
        self.items = nn.ModuleList([nn.Linear(8, 8) for _ in range(n)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        return iter(self.items)


class IndexedNoIter(nn.Module):
    """Indexes and counts, but leaves iteration to Python's ``__getitem__``
    protocol."""

    def __init__(self, n: int = 3):
        super().__init__()
        self.items = nn.ModuleList([nn.Linear(8, 8) for _ in range(n)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class IndexesATensor(nn.Module):
    """``__getitem__`` that means something other than a submodule."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3))

    def __getitem__(self, index):
        return self.w[index]


class IndexesAFreshModule(nn.Module):
    """``__getitem__`` returning a module this tree does not wrap."""

    def __getitem__(self, index):
        return nn.Linear(8, 8)


class Container(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.seq = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.dct = nn.ModuleDict({"a": nn.Linear(8, 8)})
        self.tset = TranscoderSet()
        self.noiter = IndexedNoIter()
        self.tensor_indexed = IndexesATensor()
        self.fresh = IndexesAFreshModule()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.tset[1](x)


class TestContainerDunders:
    """``len``, iteration and indexing answer for the same entries.

    Torch's containers hold their modules as their own entries, so all three go
    through the envoy children. A container holding them one level down counts
    and indexes the modules, and the envoy follows it there rather than
    reporting a length whose every index raises.
    """

    @pytest.fixture
    def envoy(self):
        return Envoy(Container())

    def test_torch_containers_index_by_child_name(self, envoy):
        assert envoy.layers[1] is getattr(envoy.layers, "1")
        assert envoy.seq[0]._module is envoy._module.seq[0]
        assert envoy.dct["a"]._module is envoy._module.dct["a"]

    def test_torch_containers_agree_across_dunders(self, envoy):
        for child in (envoy.layers, envoy.seq, envoy.dct):
            assert len(child) == len(list(child))

    def test_a_delegating_container_agrees_across_dunders(self, envoy):
        assert len(envoy.tset) == 4
        assert len(list(envoy.tset)) == 4
        assert [child.path for child in envoy.tset] == [
            f"model.tset.items.{index}" for index in range(4)
        ]

    def test_indexing_a_delegating_container_names_the_module_it_returns(self, envoy):
        assert envoy.tset[2] is envoy.tset.items[2]
        assert envoy.tset[2]._module is envoy._module.tset[2]

    def test_a_negative_index_delegates(self, envoy):
        assert envoy.tset[-1] is envoy.tset.items[3]

    def test_iteration_is_bounded_by_len_without_dunder_iter(self, envoy):
        assert len(envoy.noiter) == 3
        assert [child.path for child in envoy.noiter] == [
            f"model.noiter.items.{index}" for index in range(3)
        ]

    def test_a_non_module_index_raises(self, envoy):
        # Handing back the tensor element would break the contract that indexing
        # an envoy yields an envoy.
        with pytest.raises(AttributeError):
            envoy.tensor_indexed[0]

    def test_an_untracked_module_raises(self, envoy):
        with pytest.raises(AttributeError):
            envoy.fresh[0]

    def test_a_delegating_container_traces(self, envoy):
        x = torch.randn(1, 8)
        with envoy.trace(x):
            second = envoy.tset[1].output.save()
        module = envoy._module
        expected = x
        for layer in module.layers:
            expected = layer(expected)
        assert torch.allclose(second, module.tset[1](expected))
