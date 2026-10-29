"""Traces run on a remote nnsight-serve engine — ``model.trace(..., serve=url)``.

A server subprocess holds the dispatched vLLM engine; the test process is the
client, writing traces against a meta model (no weights) and sending them over
HTTP. Skipped unless a GPU, ``fastapi``, and ``psutil`` are available.
"""

import os
import socket
import subprocess
import sys
import time

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("fastapi")
psutil = pytest.importorskip("psutil")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _kill_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    procs = parent.children(recursive=True) + [parent]
    for proc in procs:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)


@pytest.fixture(scope="module")
def serve_url():
    if torch.cuda.device_count() < 1:
        pytest.skip("serve needs a GPU")
    import httpx

    port = _free_port()
    env = dict(os.environ)
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nnsight.modeling.vllm.serve.cli",
            "gpt2",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            "0.1",
        ],
        env=env,
    )
    url = f"http://127.0.0.1:{port}"

    ready = False
    for _ in range(120):
        if proc.poll() is not None:
            break
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(2)

    if not ready:
        _kill_tree(proc.pid)
        pytest.skip("nnsight-serve did not become ready")

    yield url
    _kill_tree(proc.pid)


@pytest.fixture(scope="module")
def serve_client():
    """A meta client — no weights; the server holds the engine."""
    from nnsight.modeling.vllm import VLLM

    return VLLM("gpt2")


class TestGpulessClient:
    def test_meta_client_builds_without_gpu(self):
        # A serve client is CPU-only: it builds the meta tree and serializes traces,
        # never running a forward. Simulate a GPU-less host and confirm the tree and
        # tokenizer still come up.
        from unittest import mock

        from nnsight.modeling.vllm import VLLM

        # `max_memory_allocated` too (vLLM's loader reports peak memory around the
        # load): on a host that really has no GPU it returns 0, but here CUDA may
        # already be initialized by an earlier test in this process, and it then
        # rejects the device index the mocked count implies.
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "torch.cuda.device_count", return_value=0
        ), mock.patch("torch.cuda.max_memory_allocated", return_value=0):
            model = VLLM("gpt2")

        assert model.tokenizer is not None
        assert hasattr(model, "transformer")


class TestServe:
    @torch.no_grad()
    def test_basic_save(self, serve_client, serve_url, ET_prompt):
        with serve_client.trace(ET_prompt, temperature=0.0, top_p=1, serve=serve_url):
            logits = serve_client.logits.save()

        assert serve_client.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_intervention(self, serve_client, serve_url, ET_prompt):
        with serve_client.trace(ET_prompt, temperature=0.0, top_p=1, serve=serve_url):
            serve_client.transformer.h[-2].mlp.output = torch.zeros_like(
                serve_client.transformer.h[-2].mlp.output
            )
            logits = serve_client.logits.save()

        # Zeroing a late MLP on the server moves the prediction off the clean answer.
        assert serve_client.tokenizer.decode(logits.argmax(dim=-1)) != " Paris"

    @torch.no_grad()
    def test_multi_invoke(self, serve_client, serve_url, ET_prompt, MSG_prompt):
        # Two invokes in one serve trace run concurrently on the server (asyncio.gather);
        # each save comes home to its own name.
        with serve_client.trace(temperature=0.0, top_p=1, serve=serve_url) as tracer:
            with tracer.invoke(ET_prompt):
                a = serve_client.logits.save()
            with tracer.invoke(MSG_prompt):
                b = serve_client.logits.save()

        assert serve_client.tokenizer.decode(a.argmax(dim=-1)) == " Paris"
        assert serve_client.tokenizer.decode(b.argmax(dim=-1)) == " New"

    @torch.no_grad()
    def test_error_surfaces(self, serve_client, serve_url, ET_prompt):
        # An intervention error on the server comes back and raises at the client.
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            with serve_client.trace(
                ET_prompt, temperature=0.0, top_p=1, serve=serve_url
            ):
                _ = 1 / 0
                serve_client.logits.save()

    @torch.no_grad()
    def test_engine_survives_error(self, serve_client, serve_url, ET_prompt):
        # A failed serve trace leaves the shared server engine healthy for the next one.
        try:
            with serve_client.trace(
                ET_prompt, temperature=0.0, top_p=1, serve=serve_url
            ):
                _ = 1 / 0
                serve_client.logits.save()
        except RuntimeError:
            pass

        with serve_client.trace(ET_prompt, temperature=0.0, top_p=1, serve=serve_url):
            logits = serve_client.logits.save()
        assert serve_client.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"


class TestServeResult:
    @torch.no_grad()
    def test_result_comes_back(self, serve_client, serve_url, ET_prompt):
        # The server hands the finished RequestOutput to the collect, so saving
        # `tracer.result` is how a serve client reads generated tokens.
        with serve_client.trace(
            ET_prompt, temperature=0.0, top_p=1, max_tokens=3, serve=serve_url
        ) as tracer:
            result = tracer.result.save()

        assert type(result).__name__ == "RequestOutput"
        assert len(result.outputs[0].token_ids) == 3


class TestServeEdit:
    """`model.edit(serve=url)` — the block installs over HTTP, on the server's engine."""

    @torch.no_grad()
    def test_edit_runs_for_a_served_trace(self, serve_client, serve_url, ET_prompt):
        with serve_client.edit(serve=serve_url) as (tracer, edit):
            hidden = serve_client.transformer.h[5].output.save()
        try:
            with serve_client.trace(
                ET_prompt, temperature=0.0, top_p=1, max_tokens=1, serve=serve_url
            ) as tracer:
                result = tracer.result.save()

            # The edit's value rides the output of the request it ran on.
            assert "hidden" in result.saves
        finally:
            edit.clear()

    @torch.no_grad()
    def test_clear_stops_it(self, serve_client, serve_url, ET_prompt):
        with serve_client.edit(serve=serve_url) as (tracer, edit):
            hidden = serve_client.transformer.h[5].output.save()
        edit.clear()

        with serve_client.trace(
            ET_prompt, temperature=0.0, top_p=1, max_tokens=1, serve=serve_url
        ) as tracer:
            result = tracer.result.save()

        assert "hidden" not in getattr(result, "saves", {})

    @torch.no_grad()
    def test_the_client_stays_gpuless(self, serve_client, serve_url):
        # Installing must not dispatch an engine on the client — it has no weights.
        with serve_client.edit(serve=serve_url) as (tracer, edit):
            hidden = serve_client.transformer.h[5].output.save()
        try:
            assert not serve_client.dispatched
        finally:
            edit.clear()


class TestNamedEditsOverServe:
    def test_named_edit_selected_by_a_served_trace(self, serve_client, serve_url, ET_prompt):
        model = serve_client
        with model.edit(serve=serve_url, name="a") as (tracer, a):
            value_a = model.transformer.h[3].output[-1].norm().save()
        with model.edit(serve=serve_url, name="b") as (tracer, b):
            value_b = model.transformer.h[4].output[-1].norm().save()
        try:
            with model.trace(ET_prompt, serve=serve_url, max_tokens=1, edits=["a"]) as tracer:
                result = tracer.result.save()
            assert "value_a" in result.saves and "value_b" not in result.saves

            # The worker's check: it comes back as the request's deferred error,
            # re-raised here as a RuntimeError carrying the original type.
            with pytest.raises(RuntimeError, match="ValueError: edits=.*no edit is installed under that name"):
                with model.trace(ET_prompt, serve=serve_url, max_tokens=1, edits=["nope"]) as tracer:
                    result = tracer.result.save()
        finally:
            a.clear()
            b.clear()
