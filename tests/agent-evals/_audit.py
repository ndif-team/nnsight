"""Reference-solution audit for existing agent-eval tasks.

Runs every registered task through the regular `runner.run_task` entry
point with a hand-written canonical solution. Anything that fails
identifies a prompt/verify mismatch with the current refactor/transform
branch — the goal here is to catch eval suite drift, NOT to evaluate an
agent.

Usage:
    cd tests/agent-evals
    python _audit.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from tasks import list_all_tasks  # noqa: E402
from runner import run_task  # noqa: E402


# Hand-written canonical solutions for every existing task.
# Keys are task IDs; values are the body that the agent would have produced.
REFERENCE_SOLUTIONS: dict[str, str] = {
    # --- BASIC ---
    # NOTE on transformers >= 5.0: GPT2Block.forward now returns a plain
    # Tensor (not a tuple). So `model.transformer.h[i].output` is a Tensor
    # of shape [batch, seq, hidden] directly — no [0] indexing needed.
    "basic_01_trace_and_save": """
with model.trace("Hello world"):
    hidden_states = model.transformer.h[-1].output.save()
""",
    "basic_02_logits_and_prediction": """
with model.trace("The capital of France is"):
    logits = model.lm_head.output.save()
    predicted_token = logits[0, -1, :].argmax(dim=-1).save()
""",
    "basic_03_zero_activations": """
with model.trace("Hello"):
    model.transformer.h[0].output[:] = 0
    zeroed_output = model.transformer.h[0].output.save()
    logits = model.lm_head.output.save()
""",
    "basic_04_access_input": """
with model.trace("Machine learning"):
    layer_input = model.transformer.h[5].input.save()
""",
    "basic_05_clone_before_modify": """
with model.trace("Test"):
    before = model.transformer.h[0].output.clone().save()
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()
""",
    "basic_06_cache_explicit_modules": """
with model.trace("Caching test") as tracer:
    cache = tracer.cache(modules=[model.transformer.h[2], model.transformer.h[7]])

cached_h2 = cache['model.transformer.h.2'].output
cached_h7 = cache['model.transformer.h.7'].output
""",
    "basic_07_modify_input": """
import torch
with model.trace("Hello world"):
    orig = model.transformer.h[3].input.clone()
    model.transformer.h[3].input = torch.zeros_like(orig)
    layer3_output = model.transformer.h[3].output.save()
""",

    # --- INTERMEDIATE ---
    "intermediate_01_multiple_invokers": """
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        logits_1 = model.lm_head.output[:, -1].save()
    with tracer.invoke("World"):
        logits_2 = model.lm_head.output[:, -1].save()
""",
    "intermediate_02_activation_patching": """
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[:, -1, :]
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[:, -1, :] = clean_hs
        patched_logits = model.lm_head.output.save()
""",
    "intermediate_03_generation": """
with model.generate("Once upon a time", max_new_tokens=5):
    generated_tokens = model.generator.output.save()
""",
    "intermediate_04_iter_generation": """
with model.generate("Hello", max_new_tokens=3) as tracer:
    step_tokens = list().save()
    for step in tracer.iter[:3]:
        step_tokens.append(model.lm_head.output[0, -1, :].argmax(dim=-1))
""",
    "intermediate_05_gradients": """
with model.trace("Hello world"):
    hs = model.transformer.h[5].output
    hs.requires_grad_(True)
    logits = model.lm_head.output
    loss = logits.sum()
    with loss.backward():
        hidden_grad = hs.grad.save()
""",
    "intermediate_06_promptless_invoke": """
with model.trace() as tracer:
    with tracer.invoke("A"):
        pass
    with tracer.invoke("B"):
        pass
    with tracer.invoke("C"):
        pass
    with tracer.invoke():
        combined_logits = model.lm_head.output.save()
""",
    "intermediate_07_conditional_generation": """
with model.generate("Hello", max_new_tokens=5) as tracer:
    step_predictions = list().save()
    for step_idx in tracer.iter[:5]:
        if step_idx == 2:
            model.transformer.h[0].output[:] = 0
        step_predictions.append(model.lm_head.output[0, -1, :].argmax(dim=-1))
""",
    "intermediate_08_per_step_capture": """
with model.generate("Hello", max_new_tokens=3) as tracer:
    saves = list().save()
    for step in tracer.iter[:3]:
        saves.append(model.transformer.h[-1].output)
hs_step0, hs_step1, hs_step2 = saves[0], saves[1], saves[2]
""",
    "intermediate_09_tracer_result": """
with model.trace("The Eiffel Tower is in the city of") as tracer:
    trace_result = tracer.result.save()
""",
    "intermediate_10_source_tracing": """
with model.trace("Hello world"):
    attn_op_out = model.transformer.h[0].attn.source.attention_interface_0.output.save()
""",
    "intermediate_11_empty_invoke_post_iter": """
with model.generate(max_new_tokens=4) as tracer:
    with tracer.invoke("Hello"):
        step_logits = list().save()
        for step in tracer.iter[:4]:
            step_logits.append(model.lm_head.output[0, -1, :].argmax(dim=-1))
    with tracer.invoke():
        gen_result = tracer.result.save()
""",
    "intermediate_12_bounded_iter_slice": """
with model.generate("Hello", max_new_tokens=5) as tracer:
    mid_step_tokens = list().save()
    for step in tracer.iter[1:3]:
        mid_step_tokens.append(model.lm_head.output[0, -1, :].argmax(dim=-1))
""",

    # --- ADVANCED ---
    "advanced_01_sessions": """
with model.session() as session:
    with model.trace("The weather is"):
        hs = model.transformer.h[3].output.save()
    with model.trace("The climate is"):
        model.transformer.h[3].output[:] = hs
        session_logits = model.lm_head.output.save()
""",
    "advanced_02_model_editing": """
with model.edit() as model_edited:
    model.transformer.h[0].output[:] = 0

with model_edited.trace("Test"):
    edited_output = model_edited.transformer.h[0].output.save()
""",
    "advanced_03_caching": """
with model.trace("Hello world") as tracer:
    cache = tracer.cache()

cached_layer5 = cache['model.transformer.h.5'].output
""",
    "advanced_04_skip_module": """
with model.trace("Hello"):
    layer0_out = model.transformer.h[0].output
    model.transformer.h[1].skip(layer0_out)
    skipped_output = model.transformer.h[1].output.save()
""",
    # (transformers 5+: block.output is a tensor; skip(tensor) works directly.)
    "advanced_05_scan_mode": """
import nnsight
with model.scan("Hello world"):
    hidden_dim = nnsight.save(model.transformer.h[-1].output.shape[-1])
""",
    "advanced_06_barrier_sync": """
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("Paris is the capital of"):
        paris_emb = model.transformer.wte.output
        barrier()
    with tracer.invoke("_ _ _ _ _"):
        barrier()
        model.transformer.wte.output = paris_emb
        barrier_logits = model.lm_head.output.save()
""",
    "advanced_07_logit_lens": """
with model.trace("The Eiffel Tower is in"):
    layer_predictions = list().save()
    for i in range(6):
        hs = model.transformer.h[i].output
        logits = model.lm_head(model.transformer.ln_f(hs))
        layer_predictions.append(logits[0, -1, :].argmax(dim=-1))
""",
    "advanced_08_steering_vector": """
import torch
with model.trace("I think that"):
    direction = torch.randn(768).to(model.transformer.h[10].output.device)
    model.transformer.h[10].output[:, -1, :] += direction * 0.1
    steered_logits = model.lm_head.output.save()
""",
    "advanced_09_early_stop": """
with model.trace("Hello world") as tracer:
    early_output = model.transformer.h[0].output.save()
    tracer.stop()
""",
    "advanced_10_logit_lens_subset": """
with model.trace("The Eiffel Tower is in"):
    subset_predictions = list().save()
    for L in [3, 6, 9, 11]:
        hs = model.transformer.h[L].output
        logits = model.lm_head(model.transformer.ln_f(hs))
        subset_predictions.append(logits[0, -1, :].argmax(dim=-1))
""",
    "advanced_11_steering_during_generation": """
import torch
direction = torch.randn(768)
with model.generate("I think that", max_new_tokens=4) as tracer:
    d = direction.to(model.transformer.h[6].output.device)
    model.transformer.h[6].output[:, -1, :] += d * 0.1
    gen_result = tracer.result.save()
""",
    "advanced_12_custom_envoy": """
import torch
from nnsight import NNsight
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import eproperty
from nnsight.intervention.hooks import requires_output

class DoubledEnvoy(Envoy):
    @eproperty(key='output')
    @requires_output
    def doubled(self): ...

    @doubled.preprocess
    def doubled(self, value):
        return value * 2

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2),
)

model = NNsight(net, envoys={torch.nn.Linear: DoubledEnvoy})

with model.trace(torch.rand(1, 5)):
    raw = model[0].output.save()
    dbl = model[0].doubled.save()
""",
    "advanced_13_inplace_edit_clear": """
with model.edit(inplace=True):
    model.transformer.h[2].output[:] = 0

with model.trace("Hello"):
    edited_out = model.transformer.h[2].output.save()

model.clear_edits()

with model.trace("Hello"):
    cleared_out = model.transformer.h[2].output.save()
""",
    "advanced_14_python_conditional": """
import torch
with model.trace("Hello world"):
    hs0 = model.transformer.h[0].output
    if torch.all(hs0 < 10000):
        model.transformer.h[5].output[:] = 0
    gated_output = model.transformer.h[5].output.save()
""",
}


def main():
    tasks = list_all_tasks()
    rows = []
    missing_solutions = []
    failures = []

    print(f"Auditing {len(tasks)} task(s) against hand-written reference solutions.")
    print("This loads gpt2 once per task; expect 2-3 min total.\n")

    for task in tasks:
        ref = REFERENCE_SOLUTIONS.get(task.id)
        if ref is None:
            missing_solutions.append(task.id)
            rows.append((task.id, "?", "no reference solution"))
            continue

        result = run_task(task, ref.strip())
        if result.success:
            rows.append((task.id, "PASS", ""))
        else:
            err = (result.error_message or "")[:140].replace("\n", " | ")
            rows.append((task.id, "FAIL", err))
            failures.append((task.id, result))

    width = max(len(t.id) for t in tasks)
    print()
    print(f"{'Task':<{width}}  Status  Detail")
    print(f"{'-' * width}  ------  -------")
    for tid, status, detail in rows:
        print(f"{tid:<{width}}  {status:<6}  {detail}")

    print()
    n_pass = sum(1 for _, s, _ in rows if s == "PASS")
    n_fail = sum(1 for _, s, _ in rows if s == "FAIL")
    n_missing = len(missing_solutions)
    print(f"Summary: {n_pass} pass / {n_fail} fail / {n_missing} no-ref / {len(rows)} total")

    if failures:
        print("\n=== Failure details ===")
        for tid, res in failures:
            print(f"\n--- {tid} ---")
            print(res.error_message)

    return 0 if n_fail == 0 and n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
