"""Focused, executable adversarial probes for the vLLM integration.

This is deliberately not named ``test_*.py``: it prints the raw observations a
red-team report needs and each mode owns one vLLM engine process.
"""

from __future__ import annotations

import argparse
import asyncio

import nnsight


def sync_save_collision() -> None:
    """Does a trace save overwrite a same-named registered save on result?"""
    from nnsight.modeling.vllm import VLLM

    model = VLLM("gpt2", dispatch=True, gpu_memory_utilization=0.1,
                 enable_prefix_caching=False)
    with model.edit() as (_, registration):
        same = nnsight.save("registered")
    try:
        with model.trace("The Eiffel Tower is in", max_tokens=1,
                         temperature=0.0) as tracer:
            same = nnsight.save("trace")
            result = tracer.result.save()
        print("SYNC_COLLISION local_same=", same)
        print("SYNC_COLLISION result_saves=", result.saves)
        print("SYNC_COLLISION request_count=", model.vllm_entrypoint.llm_engine
              .engine_core.collective_rpc("nnsight_request_count"))
    finally:
        registration.clear()


def sync_registration_matrix() -> None:
    """Probe registration scope, output-name collisions, and cache ordering."""
    from nnsight.modeling.vllm import VLLM

    model = VLLM("gpt2", dispatch=True, gpu_memory_utilization=0.1,
                 enable_prefix_caching=False)
    try:
        events = []
        with model.edit() as (_, shared_registration):
            events.append("ran")
            snapshot = nnsight.save(list(events))
        try:
            shared = model.generate(["One", "Two"], max_tokens=1, temperature=0.0)
            print("SYNC_SHARED_CLOSURE snapshots=",
                  [output.saves["snapshot"] for output in shared])
        finally:
            shared_registration.clear()

        with model.edit() as (_, first_registration):
            same = nnsight.save("first")
        with model.edit() as (_, second_registration):
            same = nnsight.save("second")
        try:
            collision = model.generate(["One"], max_tokens=1, temperature=0.0)[0]
            print("SYNC_TWO_REGISTRATIONS_SAME_NAME saves=", collision.saves)
        finally:
            first_registration.clear()
            second_registration.clear()

        try:
            with model.trace("One", max_tokens=1, temperature=0.0) as tracer:
                value = model.logits
                cache = tracer.cache()
        except Exception as error:
            print("SYNC_CACHE_AFTER_READ error=", type(error).__name__, str(error))
        else:
            print("SYNC_CACHE_AFTER_READ unexpectedly_succeeded=", cache)
    finally:
        # This is a synchronous engine; process exit owns its workers.
        pass


def tensor_parallel_parameter_probe() -> None:
    """Contrast a gathered activation with an un-gathered model parameter."""
    from nnsight.modeling.vllm import VLLM

    model = VLLM("Qwen/Qwen2.5-0.5B", dispatch=True, tensor_parallel_size=2,
                 gpu_memory_utilization=0.1)
    with model.trace("Hello", max_tokens=1, temperature=0.0):
        # vLLM's parallel linear returns a tuple; save its tensor payload, as the
        # integration's TP regression tests do.
        activation = model.model.layers[0].self_attn.qkv_proj.output[0].save()
        weight = model.lm_head.weight.save()
    print("TP_PARAMETER activation_shape=", tuple(activation.shape))
    print("TP_PARAMETER lm_head_shape=", tuple(weight.shape))
    print("TP_PARAMETER tokenizer_vocab=", model.tokenizer.vocab_size)


async def async_registration_and_foreign_leak() -> None:
    """Exercise untested async registration and inspect a foreign request's save."""
    from nnsight.modeling.vllm import VLLM
    from nnsight.modeling.vllm.engines.engine import merge_collected
    from vllm import SamplingParams

    model = VLLM("gpt2", mode="async", dispatch=True,
                 gpu_memory_utilization=0.1, enable_prefix_caching=False)
    engine = model.vllm_entrypoint
    registration = None
    try:
        async with model.edit() as (_, registration):
            tag = nnsight.save("registered")

        outputs = await model.generate(["The Eiffel Tower is in"], max_tokens=1,
                                       temperature=0.0)
        print("ASYNC_REGISTER generated_saves=", outputs[0].saves)

        request_ids = [f"redteamforeign{i}" for i in range(12)]

        async def foreign(request_id: str) -> None:
            async for _ in engine.generate(
                "The capital of Japan is",
                SamplingParams(max_tokens=1, temperature=0.0), request_id,
            ):
                pass

        await asyncio.gather(*(foreign(request_id) for request_id in request_ids))
        print("ASYNC_FOREIGN request_count_before_collect=",
              await engine.collective_rpc("nnsight_request_count"))
        collected = merge_collected(await engine.collective_rpc(
            "collect_nnsight", args=(request_ids, request_ids)
        ))
        print("ASYNC_FOREIGN delayed_collect_count=", len(collected))
        print("ASYNC_FOREIGN delayed_collect_tags=",
              {key: value["registered"]["tag"] for key, value in collected.items()})
    finally:
        if registration is not None:
            await registration.aclear()
        try:
            engine.shutdown()
        except Exception:
            pass
        await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=("sync-collision", "sync-matrix",
                                            "async-registration", "tp-parameter"))
    args = parser.parse_args()
    if args.probe == "sync-collision":
        sync_save_collision()
    elif args.probe == "sync-matrix":
        sync_registration_matrix()
    elif args.probe == "tp-parameter":
        tensor_parallel_parameter_probe()
    else:
        asyncio.run(async_registration_and_foreign_leak())


if __name__ == "__main__":
    main()
