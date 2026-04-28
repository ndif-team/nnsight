---
title: Contributing
one_liner: Branch conventions, the pre-PR validation suite, and where to find the team.
tags: [internals, dev]
related: [docs/developing/testing.md, docs/developing/agent-evals.md, docs/developing/architecture-overview.md]
sources: [CONTRIBUTING.md, pytest.ini, tests/conftest.py]
---

# Contributing

## What this covers

A short pointer document for contributors. The full contributing guide lives in [`CONTRIBUTING.md`](../../CONTRIBUTING.md) at the repository root — go there for development setup (clone, install, environment), project structure, and detailed test instructions. This page covers the things specific to working on this branch and the validation routine before submitting a PR.

## Architecture / How it works

### Branching conventions

The repo uses three categories of branches:

- **`main`** — release-stable. PRs target this branch unless explicitly told otherwise.
- **`dev`** — integration branch where merged work accumulates between releases.
- **`feature/<name>`** — new features. Branched from `main` (or `dev`).
- **`refactor/<name>`** — non-feature internal restructuring. Branched from `main` (or `dev`). The current branch (`refactor/transform`) is one of these.
- **`fix/<name>`** — bug fixes. Branched from `main`.

Create your branch from the latest `main` unless your work is layered on a still-open feature branch.

### Pre-PR checklist

Before opening a PR, run the standard validation suite (matches `CONTRIBUTING.md`):

```bash
pytest tests/test_lm.py tests/test_tiny.py tests/test_0516_features.py \
  tests/test_debug.py tests/test_memory_cleanup.py tests/test_multiple_wrappers.py \
  --device cpu
```

Plus, depending on what you touched:

| If you touched... | Also run... |
|-------------------|-------------|
| `src/nnsight/intervention/serialization.py` or any closure / dataclass code | `tests/test_serialization_edge_cases.py`, `tests/test_lambda_serialization.py`, `tests/test_dataclass_serialization.py`, `tests/test_local_simulation.py` |
| `src/nnsight/modeling/vllm/` | `pytest tests/test_vllm.py --tp 1` (needs GPU + vllm install) |
| `src/nnsight/modeling/diffusion.py` | `pytest tests/test_diffusion.py --device cpu` |
| `src/nnsight/modeling/vlm.py` | `pytest tests/test_vlm.py --device cpu` |
| Source extraction or `tracing/base.py` | `tests/test_source.py`, `tests/test_transform.py` |
| Hooks / interleaver | `tests/test_envoys.py`, `tests/test_memory_cleanup.py`, `tests/test_multiple_wrappers.py` |

For a quick smoke test during development:

```bash
pytest tests/test_tiny.py --device cpu -x
```

See [testing.md](./testing.md) for the full test inventory.

### Code style

The project follows the conventions in [`CONTRIBUTING.md`](../../CONTRIBUTING.md):

- Match the surrounding code's style. There is no enforced formatter.
- Keep changes focused. Don't bundle unrelated refactors into a bug fix.
- Don't add docstrings, comments, or type annotations to code you didn't change.
- Comments are for the **why** when the logic is non-obvious. The **what** should be self-evident from the code.
- Don't proactively add files (READMEs, examples, helpers) unless asked.

### What gets reviewed quickly

PRs that get merged fast tend to:

- Fix one thing or add one feature.
- Include a test that fails before the change and passes after.
- Have a one-paragraph description in the PR body explaining the why.
- Reference the issue they close (`Fixes #N`).
- Pass the validation suite locally before pushing.

### What slows things down

- Touching many files for one logical change. If you find yourself editing 15 files, consider whether some of them are independent and should be separate PRs.
- Mixing formatting changes with substantive changes.
- Adding docstrings / comments / type hints to unchanged code.
- Skipping tests with a "this isn't really testable" comment without explaining why.
- Force-pushing over review feedback (use new commits; the reviewer will squash on merge).

### Getting feedback before you write code

For larger changes, open a GitHub issue first describing the approach. The team responds quickly on Discord and the forum. If your PR has been sitting for a few days without a review, ping it on Discord rather than opening a duplicate PR.

## Key files / docs

- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — full contributing guide (setup, test commands, project structure)
- [`CLAUDE.md`](../../CLAUDE.md) — agent-friendly architectural overview
- [`NNsight.md`](../../NNsight.md) — deep technical reference
- [`pytest.ini`](../../pytest.ini) — pytest configuration + marker registration
- `tests/conftest.py` — shared fixtures and CLI flags

## Where the team is

- **GitHub issues** — [github.com/ndif-team/nnsight/issues](https://github.com/ndif-team/nnsight/issues) — bugs, feature requests, design discussions
- **Discord** — [discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7) — real-time help, design discussions, hanging out
- **Forum** — [discuss.ndif.us](https://discuss.ndif.us) — longer-form questions, tutorials, community-shared code
- **Documentation** — [nnsight.net](https://nnsight.net)

## Related

- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — the canonical contributing guide
- [testing.md](./testing.md) — full test inventory + flag reference
- [agent-evals.md](./agent-evals.md) — meta-test suite for documentation quality
- [architecture-overview.md](./architecture-overview.md) — start here for the big picture
