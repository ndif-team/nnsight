"""Streaming traces on an async vLLM engine (``VLLM(mode="async")``).

The async engine yields a RequestOutput per decode step; saved values arrive on
the finished output. AsyncLLM binds its background tasks to the event loop it is
driven from, so every test shares one module-scoped loop.
"""

import asyncio

import pytest
import torch

pytest.importorskip("vllm")


@pytest.fixture(scope="module")
def async_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def vllm_gpt2_async(async_loop):
    import os

    import psutil

    from nnsight.modeling.vllm import VLLM

    if torch.cuda.device_count() < 1:
        pytest.skip("vLLM tests need a GPU")
    before = {p.pid for p in psutil.Process(os.getpid()).children(recursive=True)}
    model = VLLM(
        "gpt2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,
        dispatch=True,
        mode="async",
    )
    # Only this engine's own subprocesses, so teardown never reaps another fixture's.
    model._child_pids = {
        p.pid for p in psutil.Process(os.getpid()).children(recursive=True)
    } - before
    yield model
    _shutdown_async_engine(model, async_loop)


def _shutdown_async_engine(model, loop):
    """Wind down the async engine without leaving a process that hangs the run.

    The engine's cleanup (cancelling its output-handler task, telling the engine
    core to stop) has to run on the loop it lives on, so drive ``shutdown()`` from
    a brief ``run_until_complete`` rather than calling it bare. Then force-kill only
    the worker subprocesses THIS engine spawned (a lone survivor blocks the run's
    exit) — never every descendant, which would reap a still-live sync engine's workers.
    """
    import psutil

    async def _graceful():
        try:
            model.vllm_entrypoint.shutdown()
        except Exception:
            pass
        await asyncio.sleep(1.0)

    try:
        loop.run_until_complete(_graceful())
    except Exception:
        pass

    survivors = [
        psutil.Process(pid)
        for pid in getattr(model, "_child_pids", ())
        if psutil.pid_exists(pid)
    ]
    for proc in survivors:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(survivors, timeout=5)


class TestAsyncEngine:
    def test_streams_multiple_outputs(self, vllm_gpt2_async, async_loop, ET_prompt):
        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                logits = vllm_gpt2_async.logits.save()

            count = 0
            async for _ in tracer.backend:
                count += 1
            assert count > 1

        async_loop.run_until_complete(run())

    def test_saves_on_finished_output(self, vllm_gpt2_async, async_loop, ET_prompt):
        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                logits = vllm_gpt2_async.logits.save()

            last = None
            async for output in tracer.backend:
                last = output

            assert last.finished
            assert "logits" in last.saves
            assert last.saves["logits"].shape[-1] == vllm_gpt2_async.tokenizer.vocab_size
            token = vllm_gpt2_async.tokenizer.decode(last.saves["logits"].argmax(dim=-1))
            assert token == " Paris"

        async_loop.run_until_complete(run())

    def test_only_last_output_is_finished(
        self, vllm_gpt2_async, async_loop, ET_prompt
    ):
        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=3
            ) as tracer:
                logits = vllm_gpt2_async.logits.save()

            outputs = []
            async for output in tracer.backend:
                outputs.append(output)

            assert len(outputs) >= 1
            assert all(not o.finished for o in outputs[:-1])
            assert outputs[-1].finished

        async_loop.run_until_complete(run())

    def test_intervention_streams(self, vllm_gpt2_async, async_loop, ET_prompt):
        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=1
            ) as tracer:
                vllm_gpt2_async.transformer.h[-2].mlp.output = torch.zeros_like(
                    vllm_gpt2_async.transformer.h[-2].mlp.output
                )
                logits = vllm_gpt2_async.logits.save()

            last = None
            async for output in tracer.backend:
                last = output

            token = vllm_gpt2_async.tokenizer.decode(last.saves["logits"].argmax(dim=-1))
            assert token != " Paris"

        async_loop.run_until_complete(run())

    def test_await_drains_to_last(self, vllm_gpt2_async, async_loop, ET_prompt):
        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=3
            ) as tracer:
                logits = vllm_gpt2_async.logits.save()

            last = await tracer.backend
            assert last.finished
            assert "logits" in last.saves

        async_loop.run_until_complete(run())

    def test_aborted_request_frees_its_worker(
        self, vllm_gpt2_async, async_loop, ET_prompt
    ):
        # A stream closed before it finishes aborts the request, which never reaches
        # the finished-output collection that frees its worker. Closing the generator
        # (what the loop does when a consumer stops iterating) frees it instead, so
        # the tracked-request count returns to zero rather than leaking the aborted one.
        async def run():
            engine = vllm_gpt2_async.vllm_entrypoint

            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=50
            ) as tracer:
                vllm_gpt2_async.logits.save()

            stream = tracer.backend.__aiter__()
            await stream.__anext__()
            counts = await engine.collective_rpc("nnsight_request_count")
            assert counts[0] == 1

            await stream.aclose()
            counts = await engine.collective_rpc("nnsight_request_count")
            assert counts[0] == 0

        async_loop.run_until_complete(run())

    def test_error_surfaces_in_stream(self, vllm_gpt2_async, async_loop, ET_prompt):
        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=1
            ) as tracer:
                _ = 1 / 0
                logits = vllm_gpt2_async.logits.save()

            with pytest.raises(RuntimeError, match="ZeroDivisionError"):
                async for _ in tracer.backend:
                    pass

        async_loop.run_until_complete(run())

    def test_streamed_output_carries_text(
        self, vllm_gpt2_async, async_loop, MSG_prompt
    ):
        # The streamed RequestOutputs carry generated text, not only saved values.
        async def run():
            with vllm_gpt2_async.trace(
                MSG_prompt, temperature=0.0, max_tokens=3
            ) as tracer:
                vllm_gpt2_async.logits.save()

            texts = [
                output.outputs[0].text
                async for output in tracer.backend
                if output.outputs
            ]
            assert texts
            assert texts[-1].strip() == "New York City"

        async_loop.run_until_complete(run())

    def test_intermediate_outputs_have_empty_saves(
        self, vllm_gpt2_async, async_loop, ET_prompt
    ):
        # The streaming contract: every output carries a saves dict, empty until the
        # finished one.
        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=3
            ) as tracer:
                logits = vllm_gpt2_async.logits.save()

            outputs = [output async for output in tracer.backend]
            assert all(o.saves == {} for o in outputs[:-1])
            assert "logits" in outputs[-1].saves

        async_loop.run_until_complete(run())

    def test_multi_token_saves_over_the_stream(
        self, vllm_gpt2_async, async_loop, MSG_prompt
    ):
        # A per-step list filled under tracer.iter comes home whole on the finished
        # output — async saves are not limited to a single value.
        async def run():
            with vllm_gpt2_async.trace(
                MSG_prompt, temperature=0.0, max_tokens=3
            ) as tracer:
                steps = list().save()
                for _ in tracer.iter[:3]:
                    steps.append(vllm_gpt2_async.logits.argmax(dim=-1))

            last = await tracer.backend
            assert vllm_gpt2_async.tokenizer.batch_decode(last.saves["steps"]) == [
                " New",
                " York",
                " City",
            ]

        async_loop.run_until_complete(run())

    def test_multi_invoke_raises(self, vllm_gpt2_async, ET_prompt, MSG_prompt):
        # Async supports one prompt per trace; several invokes is refused (the backend
        # submits a single request). Raised synchronously on the trace's __exit__.
        with pytest.raises(NotImplementedError):
            with vllm_gpt2_async.trace(temperature=0.0, max_tokens=1) as tracer:
                with tracer.invoke(ET_prompt):
                    vllm_gpt2_async.logits.save()
                with tracer.invoke(MSG_prompt):
                    vllm_gpt2_async.logits.save()

    def test_foreign_tenant_shares_the_batch(
        self, vllm_gpt2_async, async_loop, ET_prompt, MSG_prompt
    ):
        # A plain vLLM request (no nnsight payload) generating alongside an nnsight
        # trace must not throw off the trace's token spans: scope counts the foreign
        # request's tokens but runs nothing for it.
        async def run():
            from vllm import SamplingParams

            engine = vllm_gpt2_async.vllm_entrypoint

            async def foreign():
                async for _ in engine.generate(
                    MSG_prompt, SamplingParams(temperature=0.0, max_tokens=5), "foreign-1"
                ):
                    pass

            foreign_task = asyncio.ensure_future(foreign())
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                steps = list().save()
                for _ in tracer.iter[:5]:
                    steps.append(vllm_gpt2_async.logits.argmax(dim=-1))

            last = await tracer.backend
            await foreign_task
            # The trace's own prediction is unaffected by the batch-mate.
            assert vllm_gpt2_async.tokenizer.decode(last.saves["steps"][0]) == " Paris"

        async_loop.run_until_complete(run())


class TestAsyncClearEdits:
    """Clearing installed edits on an engine whose workers can only be awaited."""

    @torch.no_grad()
    def test_aclear_edits_clears_every_one(self, vllm_gpt2_async, async_loop,
                                           ET_prompt):
        model = vllm_gpt2_async

        async def run():
            async with model.edit() as (tracer, deep):
                deep_hidden = model.transformer.h[8].output.save()
            async with model.edit() as (tracer, shallow):
                shallow_hidden = model.transformer.h[2].output.save()

            assert model._installed_edits == [deep, shallow]

            await model.aclear_edits()

            assert deep.cleared and shallow.cleared
            assert model._installed_edits == []

        async_loop.run_until_complete(run())

    @torch.no_grad()
    def test_the_synchronous_form_refuses(self, vllm_gpt2_async, async_loop):
        # It would have to await the workers from outside the loop they run on.
        # Refusing beats half-clearing, or clearing nothing and saying so.
        model = vllm_gpt2_async

        async def install():
            async with model.edit() as (tracer, edit):
                hidden = model.transformer.h[5].output.save()
            return edit

        edit = async_loop.run_until_complete(install())
        try:
            with pytest.raises(NotImplementedError, match="aclear_edits"):
                model.clear_edits()
            assert not edit.cleared
        finally:
            async_loop.run_until_complete(model.aclear_edits())
