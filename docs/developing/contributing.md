---
title: Contributing
one_liner: House style, branch/commit conventions, and the pre-PR routine for this repo.
tags: [internals, dev]
related: [docs/developing/testing.md, docs/developing/architecture-overview.md]
sources: [STYLE.md, pyproject.toml, tests/]
---

# Contributing

## What this covers

How work lands in this repo: the canonical style guide, branch and commit
conventions, and what to run before opening a PR. The authoritative style document
is [`STYLE.md`](../../STYLE.md) at the repository root — read it first; the notes
here don't restate it.

## Style: read `STYLE.md`

[`STYLE.md`](../../STYLE.md) is the house style, written for humans and agents
alike. Its philosophy drives every rule, so when a rule doesn't cover your case the
philosophy tells you what to do. The load-bearing points:

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

## Deprecations

Every deprecation warns under `nnsight.NNsightDeprecationWarning`, with
`stacklevel=2` so the warning points at the caller's line:

```python
warnings.warn(
    "model.generator.output is deprecated; use tracer.result instead.",
    NNsightDeprecationWarning,
    stacklevel=2,
)
```

Three rules make a deprecation warning that people actually see:

- **The category is `NNsightDeprecationWarning`, a `FutureWarning` subclass.**
  Not `DeprecationWarning`. Python's default filters are
  `default::DeprecationWarning:__main__` followed by `ignore::DeprecationWarning`,
  so a `DeprecationWarning` raised from inside a package — which is where every
  nnsight deprecation is raised — warns to nobody. `FutureWarning` is the category
  the language reserves for deprecations addressed to a library's *users*, and no
  default filter hides it.
- **The message names the replacement**, in one shape:
  `"<what is deprecated> is deprecated; use <what to write instead> instead."`
  Nothing else. A user who reads only the message knows what to type next.
- **nnsight registers no warning filters.** A library that calls
  `warnings.simplefilter`/`filterwarnings` at import overrides the `-W` flags and
  `PYTHONWARNINGS` its user chose. The dedicated category is what lets a caller
  silence nnsight's deprecations and nothing else:
  `warnings.filterwarnings("ignore", category=nnsight.NNsightDeprecationWarning)`.

For a whole callable, `@nnsight.deprecated("<old> is deprecated; use <new> instead.")`
wraps it and does all three. `tests/test_deprecations.py` asserts every deprecation
warns under the category, from an imported module as well as from `__main__`.

## Branches

Base PRs on the active release branch — `0.8` — not `main`. Development happens on
the release branch, and `.github/workflows/python-app.yml` tests pushes to `main`
and `0.8` only, so a branch cut from anywhere else is untested. `main` is
release-stable. Interactive git flags (`git rebase -i`, `git add -i`)
aren't used here — prefer new commits over force-pushes during review; the
reviewer squashes on merge.

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

There is no separate `CONTRIBUTING.md`; the routine is the test suite.

Run the offline suite on CPU (see [testing.md](./testing.md) for the details):

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ --ignore=tests/vllm --ignore=tests/tp
```

Drop `--ignore=tests/tp` on transformers >= 5.16; below it the tensor-parallel
tests are red for a reason that has nothing to do with your change (see
[testing.md](./testing.md)).

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
| `modeling/transformers.py`, `modeling/huggingface.py` | `tests/test_language.py`, `tests/test_modeling.py`, `tests/test_encoder.py`, `tests/test_chunked_tasks.py`, `tests/test_construction_routing.py` |
| `modeling/diffusion.py` | `tests/test_diffusion.py` |
| `modeling/vlm.py` | `tests/test_vlm.py` |
| `modeling/quantization.py` | `tests/test_quantization.py` |
| `modeling/tp/` | `tests/test_tensor_parallel_rules.py`, `python -m pytest tests/tp/` (needs transformers >= 5.16) |
| anything you deprecate | `tests/test_deprecations.py` |
| `intervention/backends/remote.py` | `tests/test_remote_backend.py`, `tests/test_serialization.py` (`remote="local"`) |
| `modeling/vllm/` | `python -m pytest tests/vllm/` (needs GPU + `vllm`) |

A PR merges fastest when it fixes or adds one thing, ships a test that fails before
and passes after, explains the why in a paragraph, and passes the suite locally.

## Where the team is

- **GitHub** — issues and PRs on the nnsight repository.
- **Discord** — [discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7) — real-time help and design discussion.
- **Forum** — [discuss.ndif.us](https://discuss.ndif.us) — longer-form questions.
- **Docs** — [nnsight.net](https://nnsight.net).

## Related

- [`STYLE.md`](../../STYLE.md) — the canonical style guide
- [testing.md](./testing.md) — full test inventory and how to run offline
