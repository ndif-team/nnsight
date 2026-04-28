---
title: Agent Evaluations
one_liner: tests/agent-evals/ — a Claude/GPT-driven harness that measures whether AI agents can write correct nnsight code from the docs.
tags: [internals, dev]
related: [docs/developing/testing.md, docs/developing/contributing.md]
sources: [tests/agent-evals/README.md, tests/agent-evals/eval.py, tests/agent-evals/agent.py, tests/agent-evals/runner.py, tests/agent-evals/run_agent_session.py, tests/agent-evals/generate_prompts.py, tests/agent-evals/tasks/]
---

# Agent Evaluations

## What this covers

`tests/agent-evals/` is a meta-test suite. Instead of testing nnsight directly, it tests whether an LLM agent (Claude, GPT, etc.) can read the nnsight documentation and produce working nnsight code. The result is a pass-rate signal that doubles as a documentation-quality metric: if an agent fails a task, either the agent is too small or the docs are too unclear about that pattern.

This is meta-recursive in an obvious way — the agent reading **this** doc is exactly the kind of agent the eval is designed to test.

## Architecture / How it works

### What it does

1. Loads a documentation bundle (typically `CLAUDE.md` plus referenced docs).
2. Presents a task prompt at one of three difficulty tiers (basic / intermediate / advanced).
3. Gets the agent to produce nnsight code for the task.
4. Executes the code against a real model.
5. Runs a per-task verifier function on the resulting locals dict.
6. Reports pass/fail per task plus aggregate pass rate.

### Components

`tests/agent-evals/agent.py`
- `AgentConfig` dataclass (provider, model, temperature, max_tokens, api_key).
- `Agent` abstract base class with `generate_code(task_prompt, documentation)`.
- `AnthropicAgent`, `OpenAIAgent` concrete implementations using the respective SDKs.
- `create_agent(config)` factory.

`tests/agent-evals/eval.py`
- Main CLI entry point. Drives an `Agent`, runs every task (or a filtered subset), and writes results to a JSON file.
- Supports `--provider`, `--model`, `--task-id`, `--difficulty`, `--output`, `--verbose`, `--list-tasks`.
- Aggregates `EvalResult` with per-difficulty breakdowns.

`tests/agent-evals/runner.py`
- `run_task(task, code)` — executes the agent's code against the model fixture, captures locals, runs `task.verify(...)`, returns a `TaskResult` (success, verification_passed, code_executed, error if any).
- `extract_code_from_response(text)` — parses code fences out of an agent's reply.

`tests/agent-evals/run_agent_session.py`
- Interactive harness for testing agents that aren't behind an API (e.g. Claude Code in Cursor). Presents tasks one at a time, you paste responses, results are saved.

`tests/agent-evals/generate_prompts.py`
- Generates a directory of standalone Markdown task prompts (`prompts/task_NN_*.md`) suitable for pasting into an external chat UI.

`tests/agent-evals/eval_responses.py`
- Replay mode: takes a JSON file of pre-recorded responses (e.g. from manual interaction with Claude Code) and runs the verification pipeline.

`tests/agent-evals/human_eval.py`
- Lets a human play the agent role for calibration.

`tests/agent-evals/tasks/`
- `__init__.py` — exports `TASKS`, `get_task`, `get_tasks_by_difficulty`, `list_all_tasks`.
- `registry.py` — `Task` dataclass + `Difficulty` enum + `register_task` decorator.
- `basic/`, `intermediate/`, `advanced/` — each has an `__init__.py` that registers tasks for that tier.

`tests/agent-evals/prompts/`
- Pre-generated task prompts (`task_01_basic_01_trace_and_save.md` through `task_19_advanced_08_steering_vector.md`).
- `00_INSTRUCTIONS.md` — meta-prompt for agents.
- `responses_template.json` — schema for recording responses.

### Task tiers

19 tasks total, from `tests/agent-evals/README.md`:

**Basic (5):** trace and save, logits and prediction, zero out activations (in-place), access module inputs, clone-before-modify pattern.

**Intermediate (6):** multiple invokers for batching, activation patching across invokes, multi-token generation, iterating over generation steps, gradient access via `with tensor.backward():`, prompt-less invokers for batch-wide ops.

**Advanced (8):** sessions, model editing, activation caching, module skipping, scan mode for shape discovery, barrier synchronization, logit lens implementation, steering vector application.

### Running an eval

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=...
python tests/agent-evals/eval.py --provider anthropic --model claude-sonnet-4-20250514 --verbose

# OpenAI GPT
export OPENAI_API_KEY=...
python tests/agent-evals/eval.py --provider openai --model gpt-4o --verbose

# Single task
python tests/agent-evals/eval.py --task-id basic_01_trace_and_save --verbose

# All tasks of a difficulty
python tests/agent-evals/eval.py --difficulty intermediate --verbose

# List tasks
python tests/agent-evals/eval.py --list-tasks

# Save full result JSON
python tests/agent-evals/eval.py --output results.json --verbose
```

Interactive (paste responses from an external agent like Claude Code):

```bash
python tests/agent-evals/run_agent_session.py --difficulty basic
```

### Result schema

```json
{
  "timestamp": "...",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "summary": {
    "total_tasks": 19,
    "passed_tasks": 15,
    "failed_tasks": 4,
    "pass_rate": 0.789
  },
  "by_difficulty": {
    "basic": {"passed": 5, "total": 5},
    "intermediate": {"passed": 5, "total": 6},
    "advanced": {"passed": 5, "total": 8}
  },
  "task_results": [
    {"task_id": "basic_01_trace_and_save", "success": true, "verification_passed": true, "code_executed": "..."},
    ...
  ]
}
```

### Adding a new task

From `tests/agent-evals/README.md`:

1. Pick the right difficulty folder under `tasks/`.
2. Define `TASK_XX_PROMPT`, `TASK_XX_SETUP`, and a `verify_task_xx(result: dict) -> bool` function.
3. `register_task(Task(id=..., name=..., difficulty=..., prompt=..., setup_code=..., verify=..., expected_output_description=..., tags=[...]))`.

`setup_code` is exec'd to define `model` (and optional helpers) before the agent's code runs. The agent's code is also exec'd into the same namespace, and the resulting locals dict is passed to `verify`.

### Reading the pass rate

From `tests/agent-evals/README.md`:

| Pass rate | Interpretation |
|-----------|----------------|
| > 80% | Documentation is excellent for agent code generation |
| 60–80% | Good, but has gaps |
| 40–60% | Needs improvement for agent comprehension |
| < 40% | Significant doc improvements needed |

Per-task failures are signal for where the docs are unclear or the patterns are too subtle. Look at `code_executed` for failed tasks to see exactly what the agent thought the docs said.

## Key files / classes

- `tests/agent-evals/README.md` — full user-facing docs for the eval suite
- `tests/agent-evals/eval.py` — main CLI
- `tests/agent-evals/agent.py:9` — `AgentConfig`
- `tests/agent-evals/agent.py:27` — `Agent` abstract base
- `tests/agent-evals/runner.py` — `run_task`, `extract_code_from_response`
- `tests/agent-evals/run_agent_session.py` — interactive harness
- `tests/agent-evals/generate_prompts.py` — generate standalone Markdown task prompts
- `tests/agent-evals/eval_responses.py` — replay mode
- `tests/agent-evals/tasks/registry.py` — task registry + Difficulty enum
- `tests/agent-evals/tasks/basic/`, `intermediate/`, `advanced/` — task definitions
- `tests/agent-evals/prompts/` — pre-generated standalone prompts
- `tests/agent-evals/requirements.txt` — anthropic + openai SDK deps

## Lifecycle (one task evaluation)

1. Load `Agent` per `AgentConfig`.
2. Load `Task` from registry (`tasks.get_task(task_id)`).
3. Build documentation string (typically full `CLAUDE.md`).
4. `agent.generate_code(task.prompt, documentation)` returns text.
5. `extract_code_from_response(text)` strips fences.
6. `runner.run_task(task, code)`:
   - Exec `task.setup_code` into a fresh dict.
   - Exec the agent's code into the same dict.
   - Call `task.verify(locals_dict)`.
   - Build a `TaskResult`.
7. Aggregate into `EvalResult`.
8. Optionally write JSON.

## Extension points

- **New provider.** Subclass `Agent`, override `generate_code`. Register in `agent.create_agent`.
- **New task.** See "Adding a new task" above.
- **Different documentation bundle.** Pass a different doc string in `eval.py` — the agent is given whatever the runner passes as `documentation`.
- **Custom verifier patterns.** `task.verify` is just a callable on the post-exec locals dict. Common patterns: shape checks, equality with reference tensors, presence of expected variable names.

## Related

- [testing.md](./testing.md) — main pytest test suite
- [contributing.md](./contributing.md) — when to run agent-evals (typically not on every PR)
- `tests/agent-evals/README.md` — fuller user docs for the eval suite
