---
title: Contributing
one_liner: House style, branch/commit conventions, the pre-PR test routine, and the docs standard for this repo.
tags: [internals, dev]
related: [docs/developing/testing.md, docs/developing/architecture-overview.md]
sources: [STYLE.md, pyproject.toml, tests/]
---

# Contributing

## What this covers

How work lands in this repo: the style guide, branch and commit conventions, what
to run before opening a PR, and how to update docs. Setup, prerequisites, bug
reports and community links are in the root [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Style: read `STYLE.md`

[`STYLE.md`](../../STYLE.md) at the repository root is the authoritative house
style; when a rule doesn't cover your case, its philosophy section decides. The
load-bearing points:

- **Comment *why*, never *what*.** If a comment paraphrases the line under it,
  delete it. Say what the reader can't see: greenlet/weakref lifetimes, handoff
  ordering, autograd aliasing, serialization boundaries. The source is present
  tense — no "this used to", no issue numbers, no TODO/FIXME. Git holds history.
- **Three-layer docs.** The module docstring teaches the concept; a thing's
  docstring states its contract and reasoning; the inline comment explains why
  *this* implementation. Every core module opens with a concept-teaching docstring.
- **Delete before you add.** Prefer removing a concept to introducing one — a plain
  method over a custom descriptor, one dispatch point over a fan-out of handlers.
- **Layer in one direction:** `tracing → intervention → modeling`. Nothing in
  `tracing/` imports `intervention/` or `modeling/`; nothing imports the root
  `__init__` from inside the package.
- **The bare (non-underscore) surface is the public API.** Underscore everything
  that isn't user-facing. Extension points are underscore-prefixed methods with
  working defaults — no ABCs, no `Mixin` suffix.
- **Modern typing throughout:** `from __future__ import annotations` after every
  module docstring; `X | None`, `list[str]`, not `Optional`/`List`. Annotate every
  parameter and return.
- **Raise by default; warn only** when the program can proceed with a
  degraded-but-sane result. No `assert`, no `logging` in library code. Error
  messages are terse, quote values with `{x!r}`, and name the user-facing API
  rather than internals.

## Branches

Base PRs on `dev`; `main` is the release branch. CI
(`.github/workflows/python-app.yml`) runs on every pull request and on pushes to
`main` and `dev`. Prefer new commits over force-pushes during review; the reviewer
squashes on merge.

## Commits

Commit subjects are `area: lowercase imperative summary`, where `area` is the part
of the tree touched. From the log:

```
tracing: add mark() for out-of-trace saves; guard save() to inside a trace
transformers: keep standalone children across a model env rebind
vllm: stream traces on an async engine (mode="async")
remote: async backend returns the saves dict instead of pushing
```

The body explains the *why* and the constraint — the same standard the source
comments hold to (name the failure mode, the counterfactual). Commits made with
agent assistance end with a trailer:

```
Co-Authored-By: <agent name> <noreply@anthropic.com>
```

## Before you open a PR

Run the offline suite on CPU (details in [testing.md](./testing.md)):

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ --ignore=tests/vllm --ignore=tests/tp
```

Drop `--ignore=tests/tp` on transformers >= 5.16; below it the tensor-parallel
tests fail for reasons unrelated to your change.

Then, depending on what you touched, run the mirroring test file(s) — the suite is
laid out one file per concept:

| If you touched... | Run at least |
|-------------------|--------------|
| `intervention/serialization.py`, `schema/request.py` | `tests/test_serialization.py` |
| `intervention/source.py` | `tests/test_source.py`, `tests/test_interleaving.py` |
| `intervention/interleaver.py`, the controller | `tests/test_interleaving.py`, `tests/test_memory.py`, `tests/test_multiple_wrappers.py` |
| `tracing/` | `tests/test_tracing.py` |
| `intervention/batching.py` | `tests/test_batching.py` |
| `intervention/fragments.py` | `tests/test_fragments.py` |
| `modeling/transformers.py`, `modeling/huggingface.py` | `tests/test_language.py`, `tests/test_modeling.py`, `tests/test_encoder.py`, `tests/test_encoder_decoder.py`, `tests/test_pretokenized.py`, `tests/test_chunked_tasks.py`, `tests/test_construction_routing.py` |
| `modeling/diffusion.py` | `tests/test_diffusion.py` |
| `modeling/vlm.py` | `tests/test_vlm.py` |
| `modeling/quantization.py` | `tests/test_quantization.py` |
| `modeling/tp/` | `tests/test_tensor_parallel_rules.py`, `python -m pytest tests/tp/` (needs transformers >= 5.16) |
| anything you deprecate | `tests/test_deprecations.py` |
| `intervention/backends/remote.py` | `tests/test_remote_backend.py`, `tests/test_serialization.py` (`remote="local"`) |
| `modeling/vllm/` | `python -m pytest tests/vllm/` (needs GPU + `vllm`) |

A PR merges fastest when it fixes or adds one thing, ships a test that fails before
and passes after, explains the why in a paragraph, and passes the suite locally.

## Updating docs

Docs change in the same PR as the code that invalidates them. They are written for
a reader — often a coding agent — who finds one page and applies it, so:

- **One topic, one home.** Put new material on the page that owns the topic and link
  to it from elsewhere; never copy a section. The layout is in
  [docs/index.md](../index.md): `usage/` per feature, `patterns/` per technique,
  `errors/` keyed by what the user saw, `developing/` for internals.
- **Compact.** State the rule, show one example, stop. No restating, no history
  ("used to", "now raises") — the page describes the code as it is.
- **Nothing environment-specific**: no machine or host names, personal paths,
  tunnels, or one deployment's details. Generalize the requirement or leave it out.
- **A new page gets an index line** in its section's `index.md`.

The full standard is the Documentation section of the root
[CONTRIBUTING.md](../../CONTRIBUTING.md#documentation).
