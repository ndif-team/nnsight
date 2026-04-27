# nnsight Agent Evaluation Suite

This suite benchmarks **how well the nnsight documentation helps agents (and humans) use, guide, and develop with nnsight**. It does this two ways:

- **Code tasks** — the agent writes nnsight code; the suite executes it on a real model and verifies the result.
- **Multiple-choice questions (MCQs)** — the agent picks one of several options; the suite compares the chosen letter to the correct answer.

The suite is documentation-aware: every run feeds a chosen **doc bundle** into the agent's system prompt. By varying the bundle (minimal vs. router vs. full vs. legacy) you can measure each section's contribution to agent success — that's the documentation-benchmark dial.

## Contents

- 33 code-generation tasks (7 basic + 12 intermediate + 14 advanced)
- 32 MCQs (5 basic + 13 intermediate + 14 advanced)
- 4 doc bundles for varying the agent's documentation context

---

## Quick start: testing Claude Code in your IDE

```bash
cd tests/agent-evals

# Generate per-task .md prompts + a JSON response template
python generate_prompts.py --output prompts/

# In Claude Code (or another agent) — point it at the prompts and CLAUDE.md.
# Save the agent's responses (one entry per task) into responses.json, then:
python eval_responses.py responses.json --verbose
```

Or run an interactive paste-loop:

```bash
python run_agent_session.py --difficulty basic
```

---

## Quick start: programmatic eval against an LLM API

```bash
# Anthropic (default; uses the docs/router bundle)
export ANTHROPIC_API_KEY=...
python eval.py --provider anthropic --model claude-sonnet-4-6 --verbose

# OpenAI
export OPENAI_API_KEY=...
python eval.py --provider openai --model gpt-4o --verbose

# Single task / one difficulty / only MCQs
python eval.py --task-id basic_01_trace_and_save --verbose
python eval.py --difficulty basic --verbose
python eval.py --kind mcq --verbose

# List every registered task
python eval.py --list-tasks
```

### Documentation bundles (`--doc-bundle`)

The whole point of the suite is to score documentation. Vary the bundle to measure each section's contribution:

| Bundle | Contents | Approx. size |
|---|---|---|
| `minimal` | `CLAUDE.md` only | ~14 KB |
| `router` (default) | `CLAUDE.md` + `docs/concepts/` + `docs/gotchas/` + `docs/errors/` + `docs/reference/` | ~260 KB |
| `full` | `CLAUDE.md` + entire `docs/` tree + `README.md` + `0.6.0.md` | ~870 KB |
| `legacy` | `CLAUDE.md` + truncated `NNsight.md` (the pre-`docs/` loader) | ~65 KB |

```bash
# Run the same suite against three different bundles to compare:
python eval.py --doc-bundle minimal --output results/minimal.json
python eval.py --doc-bundle router  --output results/router.json
python eval.py --doc-bundle full    --output results/full.json

# Inspect the bundle contents:
python doc_bundles.py --bundle full | less

# Or just the byte size table:
python doc_bundles.py --size
```

A typical doc-investment study compares pass-rate across bundles for the same model — the delta is what each section is buying you.

---

## Output format

```json
{
  "timestamp": "2026-04-27T12:00:00",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "doc_bundle": "router",
  "summary": {
    "total_tasks": 65,
    "passed_tasks": 52,
    "failed_tasks": 13,
    "pass_rate": 0.800
  },
  "by_difficulty": {
    "basic": {"passed": 9, "total": 10},
    "intermediate": {"passed": 21, "total": 25},
    "advanced": {"passed": 22, "total": 30}
  },
  "by_kind": {
    "code": {"passed": 28, "total": 33},
    "mcq":  {"passed": 24, "total": 32}
  },
  "task_results": [
    {"task_id": "basic_01_trace_and_save", "success": true, ...},
    ...
  ]
}
```

The `by_kind` table is the cleanest signal. Code-task pass rate measures *production* (can the agent write working nnsight code?); MCQ pass rate measures *understanding* (does the agent know the right pattern when shown alternatives?).

---

## Task categories

### Code tasks (33)

**Basic (7)** — fundamentals
- `basic_01_trace_and_save` — write a trace block and `.save()` the output
- `basic_02_logits_and_prediction` — capture lm_head logits + argmax
- `basic_03_zero_activations` — in-place `[:] = 0`
- `basic_04_access_input` — `.input` vs `.output`
- `basic_05_clone_before_modify` — `.clone()` before in-place modification
- `basic_06_cache_explicit_modules` — `tracer.cache(modules=[...])`
- `basic_07_modify_input` — write to `.input` (replacement, not slice)

**Intermediate (12)**
- `intermediate_01_multiple_invokers` — `with tracer.invoke(...):` × N
- `intermediate_02_activation_patching` — same-module barrier pattern
- `intermediate_03_generation` — `model.generate(input, max_new_tokens=N)`
- `intermediate_04_iter_generation` — `for step in tracer.iter[:N]`
- `intermediate_05_gradients` — `with loss.backward(): ...grad`
- `intermediate_06_promptless_invoke` — empty invoke on combined batch
- `intermediate_07_conditional_generation` — `if step_idx == 2: ...`
- `intermediate_08_per_step_capture` — collect a tensor per generation step
- `intermediate_09_tracer_result` — `tracer.result.save()`
- `intermediate_10_source_tracing` — `module.source.<op>.output`
- `intermediate_11_empty_invoke_post_iter` — the canonical "code after iter" fix
- `intermediate_12_bounded_iter_slice` — `tracer.iter[1:3]`

**Advanced (14)**
- `advanced_01_sessions` — multi-trace `model.session()`
- `advanced_02_model_editing` — `with model.edit() as model_edited:`
- `advanced_03_caching` — `tracer.cache()` (all modules)
- `advanced_04_skip_module` — `module.skip(value)`
- `advanced_05_scan_mode` — `model.scan(...)` for shape discovery
- `advanced_06_barrier_sync` — `tracer.barrier(n)`
- `advanced_07_logit_lens` — `lm_head(ln_f(hs))` over many layers
- `advanced_08_steering_vector` — adding a direction at one layer
- `advanced_09_early_stop` — `tracer.stop()`
- `advanced_10_logit_lens_subset` — logit lens at specific layers
- `advanced_11_steering_during_generation` — steering across all generation steps
- `advanced_12_custom_envoy` — `envoys={Cls: MyEnvoy}` + `@hooked_output()` /
  `@eproperty(...) + @requires_output` extension
- `advanced_13_inplace_edit_clear` — `model.edit(inplace=True)` + `clear_edits()`
- `advanced_14_python_conditional` — `if torch.all(hs > 0):` inside trace

### MCQs (32)

**Basic (5)** — core mental model
- save necessity • trace-without-input • `.input` vs `.inputs` • implicit invoke • trace vs generate

**Intermediate (13)** — gotchas + canonical errors + meta
- out-of-order access • cross-invoke barriers • in-place vs replacement • tuple-vs-tensor outputs • clone-before-save • unbounded iter trailing-code • `tracer.all()` semantics • `tracer.result` vs `model.generator.output` • backward-as-separate-session • gradient reverse access order • `PYMOUNT` semantics • `barrier(n)` semantics • DEBUG mode

**Advanced (14)** — errors + extension API + integrations
- `MissedProviderError` / `OutOfOrderError` hierarchy • invoke-during-execution • recursive `.source` • eproperty mechanics • `Envoy.__call__` `hook=` default • scan + save • `edit(inplace=True)` • session bundling • remote `.save()` as transmission • blocking vs non-blocking • vLLM `.logits` / `.samples` • vLLM PP unsupported • Mediator events • logit-lens via `hook=False`

---

## CLI options

```
python eval.py [options]

Selection:
  --task-id ID            Run a specific task (repeatable)
  --difficulty {basic,intermediate,advanced}
  --kind {code,mcq}       Run only code or only MCQ tasks

Documentation:
  --doc-bundle {minimal,router,full,legacy}
                          Which docs the agent sees (default: router)
  --nnsight-path PATH     Path to the nnsight repo root (default: ../..)

Agent:
  --provider {anthropic,openai}
  --model NAME            Model name (default: claude-sonnet-4-6)
  --temperature FLOAT     Default 0.0
  --output FILE           Write JSON results to a file
  --verbose, -v           Per-task progress
  --list-tasks            Print every registered task and exit
```

---

## Adding tasks

### A new code task

```python
# in tasks/<difficulty>/__init__.py

TASK_15_PROMPT = """
Write nnsight code that ...
"""

TASK_15_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""

def verify_task_15(result: dict) -> bool:
    if "expected_var" not in result:
        return False
    return result["expected_var"].shape[-1] == 768

register_task(Task(
    id="advanced_15_my_task",
    name="Human Readable Name",
    difficulty=Difficulty.ADVANCED,
    prompt=TASK_15_PROMPT,
    setup_code=TASK_15_SETUP,
    verify=verify_task_15,
    expected_output_description="What the output should look like",
    tags=["relevant", "tags"],
))
```

After adding, write a hand-written reference solution into `_audit.py`'s `REFERENCE_SOLUTIONS` dict and run:

```bash
python _audit.py
```

This catches eval-suite drift (verify mismatches, broken canonical patterns) without burning API tokens.

### A new MCQ

```python
# in tasks/mcqs.py

register_mcq(
    id="mcq_intermediate_14_my_question",
    name="Short label",
    difficulty=Difficulty.INTERMEDIATE,
    question="What does X do when Y?",
    choices=[
        "Wrong-but-plausible distractor.",
        "The correct answer.",
        "Another wrong distractor.",
    ],
    correct_index=1,
    explanation="Cite the doc/source so failures are actionable.",
    tags=["topic", "intermediate"],
)
```

---

## Architecture

```
tests/agent-evals/
├── eval.py              # Programmatic eval (calls LLM API)
├── eval_responses.py    # Eval saved responses (manual flow)
├── run_agent_session.py # Interactive paste-loop session
├── human_eval.py        # Human review of failed tasks
├── generate_prompts.py  # Emit per-task .md prompts
├── agent.py             # Anthropic / OpenAI clients (handles MCQ + code)
├── runner.py            # Code execution + MCQ choice parsing
├── doc_bundles.py       # Doc-bundle loader (minimal/router/full/legacy)
├── _audit.py            # Reference-solution audit (no API)
└── tasks/
    ├── __init__.py
    ├── registry.py      # Task / register_task / register_mcq
    ├── basic/__init__.py
    ├── intermediate/__init__.py
    ├── advanced/__init__.py
    └── mcqs.py          # All MCQs in one file
```

The `runner.run_task(task, response)` entry point dispatches on `task.kind`:
- `CODE` → exec setup + agent code, then `task.verify(namespace)`.
- `MCQ` → parse the chosen letter (or number, or "Answer: B" form) via `parse_mcq_answer`, compare to `task.correct_index`.

`runner.parse_mcq_answer(...)` is robust to several response formats: bare letter, `Answer: B`, `(A)`, `I choose B.`, or a 1-based number.

---

## Interpreting results

| Pass rate | What it suggests |
|---|---|
| ≥ 90% | Documentation is excellent for agent code generation at this difficulty. |
| 75–90% | Good with minor gaps. Look at the small set of failures for patterns. |
| 50–75% | Significant doc gaps. Compare bundles to localize them. |
| < 50% | Doc problems dominate. Agent isn't getting enough information. |

**Code tasks vs. MCQs.** A pattern of "MCQ pass-rate ≫ code pass-rate" usually means the agent *understands* the rule but can't *operationalize* it (the docs explain things but don't give canonical templates). The reverse pattern (code ≫ MCQ) is rare and usually means the MCQ distractors are too easy.

**Bundle deltas.** The headline number is `pass(full) - pass(minimal)` — that's the net value of the docs investment. If `pass(router) ≈ pass(full)`, the extra detail outside the router subset (concepts/gotchas/errors/reference) isn't helping. If `pass(full) > pass(router)`, something in `usage/`, `models/`, `patterns/`, `remote/`, or `developing/` is doing real work.

---

## Verifying the eval suite itself

`_audit.py` runs every code task with a hand-written canonical solution and reports pass/fail without calling any LLM API. Use this whenever:

- You add a new code task (to confirm the `verify` matches the canonical solution).
- A dependency upgrade changes nnsight or transformers (we hit this exact case at the last upgrade — transformers 5+ broke `output[0]` tuple-indexing across most tasks).
- You suspect verifier drift.

```bash
python _audit.py        # 33 pass / 0 fail / 32 no-ref / 65 total
```

The "no-ref" count is just MCQs (no code reference applies).

---

## Installation

```bash
# Inside the nnsight repo
cd tests/agent-evals
pip install -r requirements.txt
```

Plus an LLM provider:

```bash
pip install anthropic   # for --provider anthropic
pip install openai      # for --provider openai
```
