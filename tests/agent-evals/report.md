# nnsight Documentation Benchmark — Report

Run date: 2026-04-27
Model: `claude-sonnet-4-6` (via the local `claude` CLI / Max subscription)
Tasks: 65 total — 33 code-generation tasks + 32 multiple-choice questions

This report measures **how well the nnsight documentation helps an LLM agent succeed at nnsight tasks**. The eval suite was run against four documentation "bundles" in two modes:

- **static**: the entire bundle is concatenated into the agent's system prompt up-front. The agent has no tools — pure text-in / text-out.
- **browse**: the agent's system prompt is just the router (`CLAUDE.md`); the agent gets a `Read` tool scoped to the bundle's directories and is expected to navigate via the router's links to specific doc files.

Static mode tests "how good is the doc content if the agent has perfect recall." Browse mode tests "do the docs work as designed — thin router with lazy fetch."

## Plots

All in `tests/agent-evals/results/`:

| File | Shows |
|---|---|
| `bundle_comparison.png` | static-mode pass rates: overall / code / MCQ per bundle |
| `bundle_comparison_diff.png` | static-mode by difficulty (basic / intermediate / advanced) |
| `bundle_comparison_browse.png` | browse-mode pass rates: overall / code / MCQ per bundle |
| `bundle_comparison_browse_diff.png` | browse-mode by difficulty |
| `bundle_comparison_static_vs_browse.png` | head-to-head static vs browse |

## Tables

### Static mode

| Bundle | Size | Overall | Code | MCQ | Basic | Intermediate | Advanced |
|---|---|---|---|---|---|---|---|
| minimal | 14 KB | 75.4% | 57.6% | 93.8% | 91.7% | 76.0% | 67.9% |
| router | 261 KB | **98.5%** | 97.0% | 100.0% | 100.0% | 100.0% | 96.4% |
| full | 868 KB | broke | — | — | — | — | — |
| legacy | 65 KB | 84.6% | 69.7% | 100.0% | 91.7% | 88.0% | 78.6% |

`full` exceeded the 200K standard context window — every call returned `API Error: Extra usage is required for 1M context`. The bundle as defined is too big to fit in standard context; this is itself a documentation-design finding.

### Browse mode

| Bundle | Read scope | Overall | Code | MCQ | Basic | Intermediate | Advanced |
|---|---|---|---|---|---|---|---|
| minimal | (no Read access) | 75.4% | 60.6% | 90.6% | 91.7% | 80.0% | 64.3% |
| router | concepts + gotchas + errors + reference | 92.3% | 100.0% | 84.4% | 91.7% | 96.0% | 89.3% |
| full | entire `docs/` tree | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

`legacy` is not defined for browse mode (there was no `docs/` tree pre-migration to navigate).

## Bundle definitions

See `tests/agent-evals/doc_bundles.py`. Summary:

- `minimal` — `CLAUDE.md` only.
- `router` — `CLAUDE.md` + `docs/concepts/` + `docs/gotchas/` + `docs/errors/` + `docs/reference/`.
- `full` — `CLAUDE.md` + entire `docs/` tree + `README.md` + `0.6.0.md`.
- `legacy` — `CLAUDE.md` + truncated `NNsight.md` (the eval loader's pre-`docs/` behavior).

In static mode bundles control what enters the system prompt. In browse mode bundles control which directories the agent's `Read` tool can access; the system prompt is always `CLAUDE.md`.

## Key findings

### 1. The docs work as designed — `browse-full` is 100%.

When the agent has the router and a `Read` tool scoped to the full `docs/` tree, **it solves every task** (33/33 code, 32/32 MCQ) — including all 14 advanced code tasks. This is the as-built architecture and it hits the bar perfectly.

### 2. The router is doing real routing work.

`browse-minimal` (router only, no `Read`) → `browse-full` (router + `Read` over `docs/`) is **+25 points**. The router's link structure successfully points the agent at the right doc files for any given task — when it can fetch them, it gets them right.

### 3. The new `docs/` architecture beats the legacy baseline by ~15 points.

`legacy` (CLAUDE.md + truncated NNsight.md, 65 KB) tops out at **84.6%**. `static-router` (CLAUDE.md + concepts/gotchas/errors/reference, 261 KB) hits **98.5%**, and `browse-full` hits 100%. The migration to a structured `docs/` tree paid off.

### 4. Browse mode catches one weakness — `browse-router` MCQs.

Same content as `static-router`, different access pattern. MCQ pass rate drops from 100% (static) to 84.4% (browse) — the 5 MCQ misses are all **MCQs whose answers live in `docs/usage/` or `docs/patterns/`**, which the router bundle doesn't include. When the agent has to fetch but those folders aren't readable, it falls back to its prior — and is sometimes confidently wrong.

This is a real signal: **`concepts/+gotchas/+errors/+reference/` is not a complete substitute for `usage/+patterns/`**. Either add MCQ-relevant content into the router-tier docs, or expand what the router bundle covers.

### 5. Browse mode is *better* than static mode for code generation.

`browse-router` code = **100% (33/33)** vs `static-router` code = 97.0% (32/33). The lazy-fetch architecture forces the agent to consult the canonical pattern when writing code. Pre-loading the bundle lets the agent skim and rely on prior knowledge; browsing makes it engage with the source-of-truth doc.

### 6. `full` static is unusable.

The full doc corpus (~870 KB ≈ 220K tokens) doesn't fit in Claude's standard 200K context window. Static-mode `full` is therefore not a usable configuration on standard accounts. Browse mode sidesteps this — the agent only reads what it needs per task.

### 7. The 75.4% floor.

Both `minimal` configurations land at exactly **75.4%**. That's the agent's pure-prior knowledge — no docs in context, no Read access. The headline number for "what does docs investment buy you over the agent's pre-trained knowledge" is therefore **+24.6 points** (75.4% → 100%) at the upper bound, with the bulk coming from the router + lazy-fetch design.

## Recommendations

Based on the findings, three concrete improvements would tighten the score:

1. **Promote `usage/` and/or `patterns/` content into the router-tier bundle** (or move some of it into reference). The 84.4% MCQ rate for `browse-router` is the cleanest signal that the bundle definition has a hole. Putting the canonical "how to do X" patterns one tier closer to the router would push browse-router MCQ from 84% closer to 100%.

2. **Address the static `full`-bundle context overflow.** Either trim the bundle (drop `developing/`?), or accept that `full` is browse-only and document that.

3. **The eval suite caught a real doc-vs-agent-prior mismatch on the first round** — the `output[0]` tx5 issue. That fix (see `7e6fb8e`) lifted browse-mode code-task pass rate dramatically. Future iterations should re-run the suite after major doc edits as a regression check.

## Reproduction

```bash
cd tests/agent-evals

# Static-mode runs (CLAUDE.md + bundle in system prompt, no tools)
bash scripts/run_bundle_study.sh

# Browse-mode runs (router in system prompt + Read tool scoped to bundle)
MODE=browse bash scripts/run_bundle_study.sh

# Plots
python scripts/plot_bundle_study.py
```

Each mode takes ~80–180 minutes of wall time (depends on browse mode's multi-turn agent calls). Both runs spend Max-subscription credits, ~260 calls per static run + ~195 calls per browse run.

## Files

- `report.md` — this file.
- `results/bundle_comparison*.png` — five plots (overall, by-difficulty, browse, browse-by-difficulty, static-vs-browse).
- `scripts/run_bundle_study.sh` — runner.
- `scripts/plot_bundle_study.py` — plotting.
- `doc_bundles.py` — bundle definitions.
- `_audit.py` — code-task audit (no API calls; canonical solutions vs verifiers).
