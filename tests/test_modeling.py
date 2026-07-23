import json

import nnsight
import pytest
import torch
import torch.nn as nn

from nnsight.intervention.envoy import Envoy
from nnsight.modeling.huggingface import HuggingFaceModel
from nnsight.modeling.transformers import TransformersModel
from nnsight.modeling.mixins.loadable import Loadable
from nnsight.modeling.mixins.meta import Meta, MetaDevice
from nnsight.modeling.mixins.remotable import Remotable
from nnsight.tracing.backend import Backend
from nnsight.util import from_import_path, to_import_path


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)

    def forward(self, x):
        return self.mlp(self.attn(x))


class Model(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def generate(self, x=None, **kwargs):
        return self(x)


def tied_model():
    """A model that shares one module across two paths."""
    shared = nn.Linear(8, 8)
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.embed = shared
    model.decoder = nn.Module()
    model.decoder.embed = shared
    model.head = nn.Linear(8, 8)
    return model, shared


class TestUpdate:
    def test_repoints_whole_tree(self):
        old = Model()
        envoy = Envoy(old)
        new = Model()
        envoy._update(new)
        # every envoy now points at the corresponding new module
        for env in envoy.modules():
            assert env._module is dict(new.named_modules())[env.path.removeprefix("model").lstrip(".")]

    def test_nested_repointed(self):
        envoy = Envoy(Model())
        new = Model()
        envoy._update(new)
        assert getattr(envoy.layers, "0").attn._module is new.layers[0].attn

    def test_no_dangling_old_refs(self):
        old = Model()
        envoy = Envoy(old)
        new = Model()
        envoy._update(new)
        assert getattr(envoy.layers, "0").mlp._module is not old.layers[0].mlp

    def test_no_double_instrument(self):
        envoy = Envoy(Model(n_layers=1))
        before = {p: len(h) for p, h in envoy.interleaver.handles.items()}
        envoy._update(Model(n_layers=1))
        after = {p: len(h) for p, h in envoy.interleaver.handles.items()}
        assert before == after
        assert all(count == 2 for count in after.values())

    def test_shared_modules_stay_shared(self):
        old, _ = tied_model()
        envoy = Envoy(old)
        new, new_shared = tied_model()
        envoy._update(new)
        assert envoy.encoder.embed._module is new_shared
        assert envoy.decoder.embed._module is new_shared
        assert envoy.encoder.embed._module is envoy.decoder.embed._module

    def test_trace_works_after_update(self):
        envoy = Envoy(Model())
        envoy._update(Model())
        captured = {}
        with envoy.trace(torch.randn(1, 8)):
            captured["out"] = getattr(envoy.layers, "0").attn.output
        assert captured["out"].shape == (1, 8)


class TestMetaDevice:
    def test_default_factory_on_meta(self):
        with MetaDevice():
            t = torch.zeros(2, 2)
        assert t.device.type == "meta"

    def test_explicit_device_overridden(self):
        with MetaDevice():
            t = torch.ones(3, device="cpu")
        assert t.device.type == "meta"

    def test_module_params_on_meta(self):
        with MetaDevice():
            linear = nn.Linear(4, 4)
        assert linear.weight.device.type == "meta"

    def test_restores_after_context(self):
        with MetaDevice():
            pass
        assert torch.zeros(1).device.type == "cpu"


class FromId(Loadable):
    def _load(self, name, n_layers=2):
        if isinstance(name, nn.Module):
            return name
        return Model(n_layers)


class TestLoadable:
    def test_module_passthrough(self):
        model = Model()
        envoy = FromId(model)
        assert isinstance(envoy, Envoy)
        assert envoy._module is model

    def test_load_builds_model(self):
        envoy = FromId("some-id", n_layers=1)
        assert isinstance(envoy._module, Model)
        assert len(envoy._module.layers) == 1

    def test_default_load_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Loadable("some-id")


class MetaModel(Meta):
    def _load_meta(self, name, n_layers=2):
        return Model(n_layers)

    def _load(self, name, n_layers=2):
        if isinstance(name, nn.Module):
            return name
        return Model(n_layers)


class TestMeta:
    def test_lazy_loads_on_meta(self):
        envoy = MetaModel("id", n_layers=2)
        assert envoy.dispatched is False
        assert all(p.device.type == "meta" for p in envoy._module.parameters())

    def test_dispatch_loads_real_weights(self):
        envoy = MetaModel("id", n_layers=2)
        envoy.dispatch()
        assert envoy.dispatched is True
        assert all(p.device.type == "cpu" for p in envoy._module.parameters())

    def test_dispatch_repoints_tree(self):
        envoy = MetaModel("id", n_layers=2)
        envoy.dispatch()
        assert getattr(envoy.layers, "0").attn._module is envoy._module.layers[0].attn

    def test_dispatch_idempotent(self):
        envoy = MetaModel("id", n_layers=1)
        envoy.dispatch()
        module = envoy._module
        envoy.dispatch()
        assert envoy._module is module

    def test_eager_dispatch(self):
        envoy = MetaModel("id", n_layers=1, dispatch=True)
        assert envoy.dispatched is True
        assert all(p.device.type == "cpu" for p in envoy._module.parameters())

    def test_module_passthrough_is_dispatched(self):
        envoy = MetaModel(Model(1))
        assert envoy.dispatched is True

    def test_trace_after_dispatch(self):
        envoy = MetaModel("id", n_layers=2)
        envoy.dispatch()
        captured = {}
        with envoy.trace(torch.randn(1, 8)):
            captured["out"] = getattr(envoy.layers, "0").mlp.output
        assert captured["out"].shape == (1, 8)

    def test_trace_auto_dispatches(self):
        envoy = MetaModel("id", n_layers=2)
        assert envoy.dispatched is False
        captured = {}
        with envoy.trace(torch.randn(1, 8)):
            captured["out"] = getattr(envoy.layers, "0").attn.output
        # tracing an undispatched model loads real weights and rebinds fn
        assert envoy.dispatched is True
        assert all(p.device.type == "cpu" for p in envoy._module.parameters())
        assert captured["out"].device.type == "cpu"
        assert torch.isfinite(captured["out"]).all()

    def test_default_load_meta_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Meta("id")

    def test_trace_false_dispatches_and_returns_output(self):
        # trace=False bypasses tracing but still dispatches (loads real weights)
        # and returns the module's output directly.
        envoy = MetaModel("id", n_layers=1)
        assert envoy.dispatched is False
        out = envoy.trace(torch.randn(1, 8), trace=False)
        assert envoy.dispatched is True
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 8)


class TestScan:
    def test_scan_reads_shapes(self):
        envoy = MetaModel("id", n_layers=2)
        captured = {}
        with envoy.scan(torch.randn(4, 8)):
            captured["out"] = getattr(envoy.layers, "0").attn.output
            captured["in"] = getattr(envoy.layers, "1").mlp.inputs[0][0]
        assert tuple(captured["out"].shape) == (4, 8)
        assert tuple(captured["in"].shape) == (4, 8)

    def test_scan_does_not_dispatch(self):
        envoy = MetaModel("id", n_layers=2)
        assert envoy.dispatched is False
        with envoy.scan(torch.randn(1, 8)):
            getattr(envoy.layers, "0").attn.output
        # scanning only propagates shapes, so real weights are never loaded
        assert envoy.dispatched is False
        assert all(p.device.type == "meta" for p in envoy._module.parameters())

    def test_scan_values_are_fake(self):
        envoy = MetaModel("id", n_layers=1)
        captured = {}
        with envoy.scan(torch.randn(2, 8)):
            captured["out"] = getattr(envoy.layers, "0").attn.output
        assert "Fake" in type(captured["out"]).__name__

    def test_scan_then_trace_still_dispatches(self):
        envoy = MetaModel("id", n_layers=1)
        with envoy.scan(torch.randn(1, 8)):
            getattr(envoy.layers, "0").attn.output
        assert envoy.dispatched is False
        with envoy.trace(torch.randn(1, 8)):
            captured = nnsight.save({})
            captured["out"] = getattr(envoy.layers, "0").attn.output
        assert envoy.dispatched is True
        assert captured["out"].device.type == "cpu"
        assert torch.isfinite(captured["out"]).all()

    def test_scan_on_dispatched_model(self):
        envoy = MetaModel("id", n_layers=1)
        envoy.dispatch()
        captured = {}
        with envoy.scan(torch.randn(3, 8)):
            captured["out"] = getattr(envoy.layers, "0").mlp.output
        # fake forward over real weights still just propagates shape
        assert tuple(captured["out"].shape) == (3, 8)
        assert "Fake" in type(captured["out"]).__name__


class RecordingBackend(Backend):
    def __init__(self):
        self.called = False

    def __call__(self, tracer):
        self.called = True
        tracer.execute(tracer.info.code)


class RemoteModel(Remotable):
    def _load_meta(self, key, n_layers=2):
        return Model(n_layers)

    def _load(self, key, n_layers=2):
        return Model(n_layers)

    def _remoteable_model_key(self):
        return f"{len(self._module.layers)}"

    @classmethod
    def _remoteable_from_model_key(cls, model_key, **kwargs):
        return cls("reconstructed", n_layers=int(model_key), **kwargs)


class TestImportPath:
    def test_round_trip(self):
        assert from_import_path(to_import_path(nn.Linear)) is nn.Linear

    def test_to_import_path_format(self):
        assert to_import_path(RemoteModel) == f"{RemoteModel.__module__}.RemoteModel"


class TestRemotable:
    def test_to_model_key(self):
        model = RemoteModel("id", n_layers=3)
        key = model.to_model_key()
        assert key == f"{RemoteModel.__module__}.RemoteModel:3"

    def test_from_model_key_round_trip(self):
        model = RemoteModel("id", n_layers=3)
        rebuilt = RemoteModel.from_model_key(model.to_model_key())
        assert isinstance(rebuilt, RemoteModel)
        assert len(rebuilt._module.layers) == 3

    def test_default_model_key_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Remotable._remoteable_model_key(object.__new__(Remotable))


class EnvModel(RemoteModel):
    """A remotable whose per-request env carries a fixed adapter id."""

    def _remoteable_get_env(self):
        return {"peft": "adapter-x"}


class TestRemoteEnv:
    def test_default_env_empty(self):
        assert RemoteModel("id")._remoteable_get_env() == {}

    def test_request_model_carries_env_through_metadata(self):
        # The env rides in the JSON envelope; a server subclass validating that
        # JSON (BackendRequestModel) inherits the field, so it round-trips.
        from nnsight.schema.request import RequestModel

        request = RequestModel(model_key="k", env={"peft": "adapter-x"})
        rebuilt = RequestModel.model_validate_json(request.metadata())
        assert rebuilt.env == {"peft": "adapter-x"}

    def test_request_model_env_defaults_empty(self):
        from nnsight.schema.request import RequestModel

        assert RequestModel(model_key="k").env == {}

    def test_remote_backend_stores_env(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        assert RemoteBackend("k", host="http://x", env={"peft": "y"}).env == {"peft": "y"}
        assert RemoteBackend("k", host="http://x").env == {}

    def test_remote_backend_stamps_env_into_request(self):
        # The RequestModel __call__ builds should carry the backend's env.
        from nnsight.intervention.backends.remote import RemoteBackend
        from nnsight.schema.request import RequestModel

        backend = RemoteBackend("k", host="http://x", env={"peft": "y"})
        captured = {}
        backend.request = lambda request, tracer: captured.update(env=request.env)
        backend(tracer=None)
        assert captured["env"] == {"peft": "y"}

    def test_trace_remote_wires_env_from_model(self):
        # trace(remote=True) sources the backend's env from _remoteable_get_env().
        from nnsight.intervention.backends.remote import RemoteBackend

        tracer = EnvModel("id").trace(torch.randn(1, 8), remote=True)
        assert isinstance(tracer.backend, RemoteBackend)
        assert tracer.backend.env == {"peft": "adapter-x"}

    def test_trace_remote_default_env_empty(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        tracer = RemoteModel("id").trace(torch.randn(1, 8), remote=True)
        assert isinstance(tracer.backend, RemoteBackend)
        assert tracer.backend.env == {}

    def test_remote_backend_blocking_defaults(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        backend = RemoteBackend("k", host="http://x")
        assert backend.blocking is True
        assert backend.job_id is None

    def test_remote_backend_non_blocking_construct(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        backend = RemoteBackend("k", host="http://x", blocking=False, job_id="abc")
        assert backend.blocking is False
        assert backend.job_id == "abc"

    def test_trace_remote_threads_blocking(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        tracer = RemoteModel("id").trace(
            torch.randn(1, 8), remote=True, blocking=False
        )
        assert isinstance(tracer.backend, RemoteBackend)
        assert tracer.backend.blocking is False

    def test_remote_backend_verbose(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        backend = RemoteBackend("k", host="http://x", verbose=True)
        assert backend.verbose is True
        assert backend.display.verbose is True

    def test_remote_backend_verbose_from_debug(self, monkeypatch):
        import nnsight
        from nnsight.intervention.backends.remote import RemoteBackend

        monkeypatch.setattr(nnsight.CONFIG.APP, "DEBUG", True)
        assert RemoteBackend("k", host="http://x").verbose is True


class FakeHF(HuggingFaceModel):
    # Override loading so the lifecycle is testable without network/transformers.
    def _load_meta(self, repo_id, *args, **kwargs):
        return Model(2)

    def _load(self, repo_id, *args, **kwargs):
        return Model(2)


class TestHuggingFace:
    def test_repo_id_and_revision(self):
        model = FakeHF("openai-community/gpt2", revision="main")
        assert model.repo_id == "openai-community/gpt2"
        assert model.revision == "main"

    def test_lazy_meta_load(self):
        model = FakeHF("some/repo")
        assert model.dispatched is False
        assert all(p.device.type == "meta" for p in model._module.parameters())

    def test_dispatch_loads_real(self):
        model = FakeHF("some/repo")
        model.dispatch()
        assert model.dispatched is True
        assert all(p.device.type == "cpu" for p in model._module.parameters())

    def test_model_key(self):
        from nnsight.modeling.huggingface import _ID_CACHE

        # Seed the canonicalization cache so the key build stays offline.
        _ID_CACHE["openai-community/gpt2"] = "openai-community/gpt2"
        model = FakeHF("openai-community/gpt2", revision="main")
        assert json.loads(model._remoteable_model_key()) == {
            "repo_id": "openai-community/gpt2",
            "revision": "main",
        }

    def test_model_key_round_trip(self):
        from nnsight.modeling.huggingface import _ID_CACHE

        _ID_CACHE["openai-community/gpt2"] = "openai-community/gpt2"
        model = FakeHF("openai-community/gpt2", revision="abc123")
        rebuilt = FakeHF.from_model_key(model.to_model_key())
        assert isinstance(rebuilt, FakeHF)
        assert rebuilt.repo_id == "openai-community/gpt2"
        assert rebuilt.revision == "abc123"

    def test_module_passthrough(self):
        # A bare module: no repo id, treated as already-loaded.
        model = HuggingFaceModel(nn.Linear(8, 8))
        assert model.repo_id is None
        assert model.dispatched is True


from transformers import BatchEncoding


class FakeTokenizer:
    def __call__(self, *inputs, return_tensors=None):
        # BatchEncoding (like a real tokenizer) so .to(device) + ** work.
        return BatchEncoding({"x": torch.randn(1, 8)})


class FakePipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = None
        self.feature_extractor = None

    def __call__(self, *inputs, **kwargs):
        return self.model(torch.randn(1, 8))

    # A trace forward leans on the pipeline to turn raw input into model inputs;
    # mirror that minimal contract (real pipelines always provide both).
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs, **kwargs):
        return self.tokenizer(inputs)


class FakeTransformers(TransformersModel):
    # Offline stand-in: skip transformers/network, drive loading ourselves.
    def _load_meta(self, repo_id, *args, **kwargs):
        if self.tokenizer is None:
            self.tokenizer = FakeTokenizer()
        model = Model(2)
        self.pipeline = FakePipeline(model, self.tokenizer)
        return model

    def _load(self, repo_id, *args, **kwargs):
        return self._load_meta(repo_id)


class TestTransformers:
    def test_typed_preprocessor_attrs(self):
        model = FakeTransformers("some/repo")
        assert model.tokenizer is not None
        assert model.processor is None
        assert model.image_processor is None
        assert model.feature_extractor is None

    def test_pipeline_attr(self):
        model = FakeTransformers("some/repo")
        assert isinstance(model.pipeline, FakePipeline)

    def test_bring_your_own_tokenizer(self):
        tok = FakeTokenizer()
        model = FakeTransformers("some/repo", tokenizer=tok)
        assert model.tokenizer is tok

    def test_meta_then_dispatch(self):
        model = FakeTransformers("some/repo")
        assert model.dispatched is False
        assert all(p.device.type == "meta" for p in model._module.parameters())
        model.dispatch()
        assert model.dispatched is True
        assert model.tokenizer is not None
        assert model.pipeline.model is model._module

    def test_trace_auto_preprocesses_string(self):
        model = FakeTransformers("some/repo")
        captured = {}
        with model.trace("hello world"):
            captured["out"] = getattr(model.layers, "0").attn.output
        assert captured["out"].shape == (1, 8)
        assert model.dispatched is True

    def test_raw_tensor_input_passes_through(self):
        model = FakeTransformers("some/repo")
        captured = {}
        with model.trace(torch.randn(1, 8)):
            captured["out"] = getattr(model.layers, "0").mlp.output
        assert captured["out"].shape == (1, 8)

    def test_generate_traces_in_with_block(self):
        model = FakeTransformers("some/repo")
        captured = {}
        with model.generate("hello"):
            captured["out"] = getattr(model.layers, "0").attn.output
        assert captured["out"].shape == (1, 8)
        assert model.dispatched is True

    def test_persistent_objects_include_pipeline_and_tokenizer(self):
        model = FakeTransformers("some/repo")
        objects = model._remoteable_persistent_objects()
        assert objects["Tokenizer"] is model.tokenizer
        assert objects["Pipeline"] is model.pipeline
        assert objects["Interleaver"] is model.interleaver
        assert "Module:model" in objects

    def test_getstate_tags_pipeline_and_tokenizer(self):
        model = FakeTransformers("some/repo")
        state = model.__getstate__()
        # The pipeline is kept (referenced by persistent id), not dropped, so a
        # deserialized model's `self.pipeline` resolves to the server's live one.
        assert state["pipeline"] is model.pipeline
        assert model.pipeline._persistent_id == "Pipeline"
        assert model.tokenizer._persistent_id == "Tokenizer"


class TestStatusDisplay:
    def _display(self, monkeypatch):
        # Force terminal (non-notebook) color rendering deterministically.
        monkeypatch.setenv("FORCE_COLOR", "1")
        from nnsight.intervention.backends.display import StatusDisplay

        display = StatusDisplay(enabled=True)
        display.notebook = False
        return display

    def _response(self, status, description=""):
        from nnsight.schema.response import ResponseModel

        return ResponseModel(id="job-1", status=status, description=description)

    def test_disabled_is_silent(self, capsys):
        from nnsight.intervention.backends.display import StatusDisplay
        from nnsight.schema.response import Status

        StatusDisplay(enabled=False).update(self._response(Status.RUNNING))
        assert capsys.readouterr().out == ""

    def test_lifecycle_then_completed(self, monkeypatch, capsys):
        from nnsight.schema.response import Status

        display = self._display(monkeypatch)
        for status in (Status.RECEIVED, Status.RUNNING, Status.COMPLETED):
            display.update(self._response(status))
        out = capsys.readouterr().out
        assert "COMPLETED" in out
        assert out.endswith("\n")  # terminal status ends the line

    def test_verbose_persists_each_status(self, monkeypatch, capsys):
        from nnsight.intervention.backends.display import StatusDisplay
        from nnsight.schema.response import Status

        monkeypatch.setenv("NO_COLOR", "1")
        display = StatusDisplay(enabled=True, verbose=True)
        display.notebook = False
        for status in (Status.RECEIVED, Status.RUNNING, Status.RUNNING, Status.COMPLETED):
            display.update(self._response(status))
        out = capsys.readouterr().out
        # One persisted line per distinct status (repeated RUNNING collapses).
        assert out.count("RECEIVED") == 1
        assert out.count("RUNNING") == 1
        assert out.count("COMPLETED") == 1
        assert out.count("\n") == 3

    def test_log_keeps_current_status(self, monkeypatch, capsys):
        from nnsight.schema.response import Status

        display = self._display(monkeypatch)
        display.update(self._response(Status.RUNNING))
        display.update(self._response(Status.LOG, "loading weights"))
        capsys.readouterr()
        # the remembered status is still RUNNING (LOG didn't replace it)
        assert display.response.status is Status.RUNNING

    def test_refresh_advances_spinner(self, monkeypatch, capsys):
        from nnsight.schema.response import Status

        display = self._display(monkeypatch)
        display.update(self._response(Status.RUNNING))
        before = display.spinner
        display.refresh()
        assert display.spinner == before + 1

    def test_log_shows_label_and_message(self, monkeypatch, capsys):
        from nnsight.schema.response import Status

        display = self._display(monkeypatch)
        display.update(self._response(Status.LOG, "loading weights"))
        out = capsys.readouterr().out
        assert "LOG" in out
        assert "loading weights" in out

    def test_error_description_not_printed_inline(self, monkeypatch, capsys):
        from nnsight.schema.response import Status

        # An ERROR's description is the full traceback; the caller re-raises it,
        # so the status line must not also print it (would show up twice).
        display = self._display(monkeypatch)
        display.update(self._response(Status.ERROR, "Traceback: kaboom"))
        out = capsys.readouterr().out
        assert "ERROR" in out
        assert "kaboom" not in out


class TestBackendThreading:
    def test_custom_backend_used(self):
        envoy = Envoy(Model())
        backend = RecordingBackend()
        with envoy.trace(torch.randn(1, 8), backend=backend):
            out = getattr(envoy.layers, "0").attn.output
        assert backend.called is True


class TestConfig:
    def test_defaults_loaded(self):
        from nnsight.schema.config import Config

        config = Config.load()
        assert config.API.HOST.startswith("http")

    def test_env_override_host(self, monkeypatch):
        from nnsight.schema.config import Config

        monkeypatch.setenv("NDIF_HOST", "https://example.test")
        assert Config.load().API.HOST == "https://example.test"

    def test_env_override_api_key(self, monkeypatch):
        from nnsight.schema.config import Config

        monkeypatch.setenv("NDIF_API_KEY", "secret")
        assert Config.load().API.APIKEY == "secret"


class TestSchema:
    def test_response_round_trip(self):
        from nnsight.schema.response import ResponseModel, Status

        response = ResponseModel(id="j1", status=Status.RUNNING, description="x")
        back = ResponseModel.unpickle(response.pickle())
        assert back.id == "j1"
        assert back.status is Status.RUNNING
        assert back.description == "x"

    def _tracer(self, block, f_locals, filename="/u/blk.py", start=1):
        """A real tracer whose traced block is ``block``, at line ``start``.

        Built without running it: parse a ``with`` block, attach its node and a
        stand-in frame — exactly the attributes serialize/deserialize touch.
        """
        import ast
        import textwrap
        import types

        from nnsight.tracing.tracer import Tracer

        src = "\n" * (start - 1) + "with ctx():\n" + textwrap.indent(block, "    ")
        node = next(
            n for n in ast.walk(ast.parse(src, filename)) if isinstance(n, ast.With)
        )
        frame = types.SimpleNamespace(
            f_locals=dict(f_locals),
            f_globals={},
            f_code=types.SimpleNamespace(
                co_filename=filename, co_firstlineno=start, co_name="<module>"
            ),
        )
        tracer = Tracer()
        tracer.info = Tracer.Info(frame, None)
        tracer.node = node
        return tracer

    def test_payload_round_trip(self):
        from nnsight.schema.request import RequestModel

        # Round-trips with and without compression; the rebuilt code runs against
        # the restored scope.
        for compress in (False, True):
            tracer = self._tracer("y = x + 1\n", {"x": 5})
            blob = RequestModel.serialize(tracer, compress=compress)
            back = RequestModel.deserialize(blob, compress=compress)

            scope = dict(back.info.frame.f_locals)
            exec(back.info.code, back.info.frame.f_globals, scope)
            assert scope["y"] == 6

    def test_deserialize_preserves_block_line_numbers(self):
        import traceback as tb_mod

        from nnsight.schema.request import RequestModel

        # `with` at line 20, `ok = 1` at 21, the raising statement at 22; the
        # rebuilt code must report line 22, not a 1-based offset into the block.
        tracer = self._tracer("ok = 1\nraise ValueError('boom')\n", {}, start=20)
        back = RequestModel.deserialize(RequestModel.serialize(tracer))

        try:
            exec(back.info.code, back.info.frame.f_globals, {})
        except ValueError as error:
            frame = tb_mod.extract_tb(error.__traceback__)[-1]
            assert frame.lineno == 22
            assert frame.filename == "/u/blk.py"
        else:
            pytest.fail("expected ValueError")

    def test_request_metadata_is_json_routing(self):
        import json

        from nnsight.schema.request import RequestModel

        meta = json.loads(RequestModel(model_key="m", session_id="s").metadata())
        assert meta == {
            "model_key": "m",
            "session_id": "s",
            "compress": False,
            "env": {},
        }


class TestSerialization:
    def _persistent_map(self, envoy):
        objects = {"Interleaver": envoy.interleaver}
        for node in envoy.modules():
            objects[f"Module:{node.path}"] = node._module
        return objects

    def test_modules_referenced_not_serialized(self):
        from nnsight.intervention.serialization import dumps, loads

        envoy = Envoy(Model())
        restored = loads(dumps(envoy), self._persistent_map(envoy))
        # modules and interleaver resolve to the same server-side objects
        assert restored._module is envoy._module
        assert getattr(restored.layers, "0").attn._module is getattr(envoy.layers, "0").attn._module
        assert restored.interleaver is envoy.interleaver
        assert [n.path for n in restored] == [n.path for n in envoy]

    def test_blob_excludes_weights(self):
        from nnsight.intervention.serialization import dumps

        # Same structure, vastly different weight sizes -> identical blob size,
        # because the modules are referenced by id, not serialized.
        small = Envoy(nn.Linear(2, 2))
        big = Envoy(nn.Linear(2000, 2000))
        assert len(dumps(small)) == len(dumps(big))

    def test_unknown_persistent_id_raises(self):
        from nnsight.intervention.serialization import (
            UnknownPersistentIdError,
            dumps,
            loads,
        )

        envoy = Envoy(Model())
        with pytest.raises(UnknownPersistentIdError):
            loads(dumps(envoy), {})  # no persistent objects provided

    def test_dump_load_file_round_trip(self):
        import io

        from nnsight.intervention.serialization import dump, load

        envoy = Envoy(Model(n_layers=1))
        buffer = io.BytesIO()
        dump(envoy, buffer)
        buffer.seek(0)
        restored = load(buffer, self._persistent_map(envoy))
        assert restored._module is envoy._module


class TestRemoteBackend:
    def test_ws_host_derivation(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        assert RemoteBackend("k", host="https://api.ndif.us").ws_host == "wss://api.ndif.us"
        assert RemoteBackend("k", host="http://localhost:5000").ws_host == "ws://localhost:5000"

    def test_invalid_host_raises(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        with pytest.raises(ValueError):
            RemoteBackend("k", host="api.ndif.us")

    def test_trace_remote_builds_backend(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        model = RemoteModel("id", n_layers=1)
        # Returns the tracer without entering it, so no connection is made.
        tracer = model.trace(torch.randn(1, 8), remote=True)
        assert isinstance(tracer.backend, RemoteBackend)
        assert tracer.backend.model_key == model.to_model_key()


class TestNNsight:
    def test_wraps_module_and_traces(self):
        from nnsight.modeling.base import NNsight

        net = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
        model = NNsight(net)
        assert isinstance(model, Envoy)
        with model.trace(torch.randn(1, 8)):
            hidden = model[0].output.save()
            out = model.output.save()
        assert tuple(hidden.shape) == (1, 16)
        assert tuple(out.shape) == (1, 4)

    def test_exported_at_top_level(self):
        import nnsight
        from nnsight.modeling.base import NNsight

        assert nnsight.NNsight is NNsight


class TestPreloadedModule:
    """A pre-loaded HF module is wrapped in a transformers.pipeline (task and
    preprocessors inferred from it), so trace/generate work — while a bare
    non-HF nn.Module through the base HuggingFaceModel wraps directly."""

    @pytest.fixture(scope="class")
    def hf_gpt2(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained("gpt2")

    @torch.no_grad()
    def test_transformers_model_from_module_traces(self, hf_gpt2):
        model = TransformersModel(hf_gpt2)
        assert model.task == "text-generation"     # inferred from the module
        assert model.tokenizer is not None         # sourced from name_or_path
        with model.trace("The Eiffel Tower is in"):
            out = model.transformer.h[0].output[0].save()
        assert isinstance(out, torch.Tensor)

    @torch.no_grad()
    def test_generate_from_module(self, hf_gpt2):
        model = TransformersModel(hf_gpt2)
        with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
            ids = tracer.result.save()
        assert ids.ndim == 2 and ids.shape[0] == 1

    def test_explicit_task_and_tokenizer_from_module(self, hf_gpt2):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        model = TransformersModel(hf_gpt2, task="text-generation", tokenizer=tok)
        assert model.tokenizer is tok

    def test_bare_module_wraps_raw(self):
        # A non-HF module through the base HuggingFaceModel is wrapped directly.
        model = HuggingFaceModel(nn.Linear(8, 8))
        assert model.dispatched is True
        assert isinstance(model._module, nn.Linear)
