"""One rank of the tensor-parallel test, run under ``torch.distributed.run``.

Not a test module (no ``test_`` prefix, so pytest does not collect it):
``test_sharded_tracing.py`` launches this, once on a single GPU for
the reference and once across N ranks, then compares what each run wrote.

It has to be a separate process because transformers tensor parallelism needs the
*calling* process to be a rank — unlike vLLM, which spawns its own workers, so
``tests/vllm/test_tensor_parallel.py`` can stay in-process.

Every rank runs the identical block. That is the whole premise of the design, so
nothing here may branch on rank.

    python -m torch.distributed.run --nproc_per_node=2 worker.py --tp 2 --out DIR
"""

from __future__ import annotations

import argparse
import os

import torch

import nnsight

PROMPT = "The Eiffel Tower is in the city of"
SECOND_PROMPT = "The capital of Japan is the city of"
LAYER = 1


def build(repo_id: str, tp: int, dtype: torch.dtype):
    from nnsight.modeling.tp import TPInterleaver
    from nnsight.modeling.transformers import TransformersModel

    if tp > 1:
        from transformers.distributed import DistributedConfig

        model = TransformersModel(
            repo_id, task="text-generation", dispatch=True, dtype=dtype,
            distributed_config=DistributedConfig(tp_size=tp),
        )
    else:
        model = TransformersModel(
            repo_id, task="text-generation", dispatch=True, dtype=dtype,
            device_map={"": 0},
        )

    # Nothing installs or enables anything: every HuggingFace model is built with
    # a TPInterleaver, which works out for itself whether it has a job to do.
    interleaver = model.interleaver
    assert isinstance(interleaver, TPInterleaver), type(interleaver)
    if tp > 1:
        assert interleaver.enabled, "sharded model did not enable the TP path"
        assert interleaver.tp_rules, "sharded model recorded no rules"
        # One interleaver serves the whole tree, including standalone children
        # outside named_children() — strand one and its values reach nothing.
        assert model.generator.interleaver is interleaver
    else:
        assert not interleaver.enabled, "unsharded model enabled the TP path"
        assert not interleaver.tp_rules

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    model = build(args.repo, args.tp, getattr(torch, args.dtype))

    layer = model.model.layers[LAYER]
    results: dict[str, torch.Tensor] = {}

    def record(name: str, tensor: torch.Tensor) -> None:
        results[name] = tensor.detach().float().cpu()

    # A column-parallel output and a row-parallel input are the two sides that
    # actually carry a shard; both must arrive at full width.
    with model.trace(PROMPT):
        record("gate_proj_out", layer.mlp.gate_proj.output.save())
        record("down_proj_in", layer.mlp.down_proj.input.save())
        record("layer_out", layer.output[0].save())
        record("baseline_logits", model.lm_head.output.save())

    # An edit straddling rank boundaries: only correct if the gather assembled in
    # true rank order and the re-split put the edit back where it came from.
    width = results["gate_proj_out"].shape[-1]
    with model.trace(PROMPT):
        layer.mlp.gate_proj.output[..., : width // 2 + 1] = 0
        record("partial_edit_logits", model.lm_head.output.save())

    # A cache must record whole tensors too, and only for what it selects.
    with model.trace(PROMPT) as tracer:
        cache = tracer.cache(modules=[layer.mlp.gate_proj], include_inputs=True).save()
    record("cached_gate_out", cache[f"model.model.layers.{LAYER}.mlp.gate_proj"].output)

    # Several invokes: the feature-dim gather has to compose with the batcher's
    # dim-0 row narrowing, which happens inside it.
    with model.trace() as tracer:
        with tracer.invoke(PROMPT):
            record("batched_first", layer.mlp.gate_proj.output.save())
        with tracer.invoke(SECOND_PROMPT):
            layer.mlp.gate_proj.output[:] = 0
            record("batched_edited_logits", model.lm_head.output.save())

    # The same sharded location revisited on every step: the occurrence tagging
    # and the gather have to stay in step across many visits.
    with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
        steps = nnsight.save([])
        for _ in tracer.iter[:3]:
            steps.append(layer.mlp.gate_proj.output[:, -1].mean())
    record("generated_steps", torch.stack([s.reshape(()) for s in steps]))

    # Greedy so the ranks cannot diverge on sampling; see the module docstring of
    # nnsight/modeling/tp/interleaver.py for why that would be a correctness bug.
    with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
        record("generated", tracer.result.save().float())

    torch.save(results, os.path.join(args.out, f"rank{rank}.pt"))


if __name__ == "__main__":
    main()
