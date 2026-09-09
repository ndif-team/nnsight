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


def build(repo_id: str, tp: int, dtype: torch.dtype, device: str = "cuda"):
    from nnsight.modeling.tp import TPFragments
    from nnsight.modeling.transformers import TransformersModel

    if tp > 1:
        from transformers.distributed import DistributedConfig

        model = TransformersModel(
            repo_id, task="text-generation", dispatch=True, dtype=dtype,
            distributed_config=DistributedConfig(tp_size=tp),
        )
    else:
        # Rank 0's card on CUDA; on CPU there is no index to pin to. The
        # reference run is single-process either way.
        model = TransformersModel(
            repo_id, task="text-generation", dispatch=True, dtype=dtype,
            device_map={"": 0} if device == "cuda" else {"": "cpu"},
        )

    # Nothing installs or enables anything: every HuggingFace model is built with
    # an ordinary interleaver carrying TPFragments, which work out for itself
    # whether they have a job to do.
    interleaver = model.interleaver
    fragments = interleaver.fragments
    assert isinstance(fragments, TPFragments), type(fragments)
    if tp > 1:
        assert fragments.enabled, "sharded model did not enable the TP path"
        assert fragments.tp_rules, "sharded model recorded no rules"
        # One interleaver serves the whole tree, including standalone children
        # outside named_children() — strand one and its values reach nothing.
        assert model.generator.interleaver is interleaver
    else:
        assert not fragments.enabled, "unsharded model enabled the TP path"
        assert not fragments.tp_rules

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    model = build(args.repo, args.tp, getattr(torch, args.dtype), args.device)

    from nnsight.modeling.tp import gather, shard

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
    # Cloned first, then assigned back: a gathered value is the output of an
    # autograd Function, which torch will not have written into in place. See
    # "Editing a gathered value" in docs/models/tensor-parallel.md.
    with model.trace(PROMPT):
        edited = layer.mlp.gate_proj.output.clone()
        edited[..., : width // 2 + 1] = 0
        layer.mlp.gate_proj.output = edited
        record("partial_edit_logits", model.lm_head.output.save())

    # Ad-hoc calls run a sharded module on whole tensors, through its own
    # hooks: a column-parallel linear (never gathered by transformers), a
    # row-parallel one (all-reduced by its post-hook) and the gathered head.
    with model.trace(PROMPT):
        mlp_in = layer.mlp.input
        down_in = layer.mlp.down_proj.input
        hidden = layer.output[0]
        record("adhoc_colwise", layer.mlp.gate_proj(mlp_in).save())
        record("adhoc_rowwise", layer.mlp.down_proj(down_in).save())
        record("adhoc_lens", model.lm_head(model.model.norm(hidden)).save())

    # An operation *inside* a module's forward is handed over as this rank's
    # piece — no rule describes one, and the axis it is split on is not knowable
    # from the value. The trace names the axis and reassembles it itself. Here
    # the column-parallel matmul is a last-dim shard of the whole.
    colwise_op = next(iter(layer.mlp.gate_proj.source.names))
    with model.trace(PROMPT):
        raw = getattr(layer.mlp.gate_proj.source, colwise_op).output
        record("source_colwise_gathered", gather(model, raw, dim=-1).save())

    # `.skip()` on a sharded module. The replacement is the caller's whole tensor
    # and never passed through the module, so there is nothing to assemble — it
    # only has to be cut back down. A row-parallel skip used to be all-reduced on
    # the way out instead: every rank read back `tp_size` times what it wrote and
    # the model carried that forward, with no error anywhere.
    with model.trace(PROMPT):
        hidden = layer.input
        layer.mlp.down_proj.skip(torch.ones_like(hidden))
        record("skip_read_back", layer.mlp.down_proj.output.save())
        record("skip_logits", model.lm_head.output.save())

    # An ad-hoc call on the *same* module whose handoff is still open. Reading
    # `down_proj.input` opens that location's visit; calling `down_proj` from
    # inside the block then asks the same location to be cut down again. The two
    # used to share one record keyed by location, so the nested call consumed the
    # open visit's and the module ran its sharded weights against a whole tensor.
    with model.trace(PROMPT):
        down_in = layer.mlp.down_proj.input
        record("adhoc_nested", layer.mlp.down_proj(down_in).save())

    # A value *between* two sharded modules is handed over as this rank's piece:
    # nothing records which axis holds the shard once it has left the module that
    # made it, and here `view`/`transpose` have moved it onto the head axis. So
    # the trace says which axis, and gathers it itself. This is the per-head
    # attention read, which is only reachable that way.
    with model.trace(PROMPT):
        heads = gather(model, layer.self_attn.source.query_states_0.output, dim=1)
        record("manual_gathered_heads", heads.save())

    with model.trace(PROMPT):
        whole = gather(model, layer.self_attn.source.query_states_0.output, dim=1).clone()
        whole[:, 1] = 0
        layer.self_attn.source.query_states_0.output = shard(model, whole, dim=1)
        record("manual_ablated_logits", model.lm_head.output.save())

    # `hook=True` only decides whether the trace watches the call; the caller is
    # holding whole tensors either way. It used to skip the bracket, so the same
    # call returned this rank's slice with the flag on and the whole without it.
    with model.trace(PROMPT):
        mlp_in = layer.mlp.input
        record("adhoc_hooked", layer.mlp.gate_proj(mlp_in, hook=True).save())

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
            layer.mlp.gate_proj.output = torch.zeros_like(layer.mlp.gate_proj.output)
            record("batched_edited_logits", model.lm_head.output.save())

    # The same sharded location revisited on every step: the occurrence tagging
    # and the gather have to stay in step across many visits.
    with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
        steps = nnsight.save([])
        for _ in tracer.iter[:3]:
            steps.append(layer.mlp.gate_proj.output[:, -1].mean())
    record("generated_steps", torch.stack([s.reshape(()) for s in steps]))

    # `backward()` through a gathered *partial* side. The re-fragment contributes
    # the whole on rank 0 and zeros elsewhere; done with a bare `zeros_like` that
    # severed the graph on every rank but 0, so the ranks stopped reaching the
    # same collectives and the backward hung — a deadlock rather than a wrong
    # number, and reached without any branch in this block.
    with model.trace(PROMPT):
        layer.mlp.down_proj.output          # forces the gather and the re-fragment
        loss = model.lm_head.output.sum()
        with loss.backward():
            record("partial_backward_grad", model.model.embed_tokens.weight.grad.save())

    # Greedy so the ranks cannot diverge on sampling; see the module docstring of
    # nnsight/modeling/tp/fragments.py for why that would be a correctness bug.
    with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
        record("generated", tracer.result.save().float())

    torch.save(results, os.path.join(args.out, f"rank{rank}.pt"))


if __name__ == "__main__":
    main()
