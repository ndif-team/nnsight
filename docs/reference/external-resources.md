---
title: External Resources
one_liner: Curated external URLs and where to go for what.
tags: [reference, links]
---

# External Resources

## Official channels

| Resource | URL | When to go there |
|----------|-----|------------------|
| Documentation site | [https://nnsight.net](https://nnsight.net) | Tutorials, API reference, conceptual guides — the primary user-facing docs. |
| GitHub repo | [https://github.com/ndif-team/nnsight](https://github.com/ndif-team/nnsight) | Source code, issues, pull requests, releases. |
| Discord | [https://discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7) | Real-time chat with the team and other users. Best for quick questions. |
| Forum | [https://discuss.ndif.us](https://discuss.ndif.us) | Long-form questions, troubleshooting, NDIF support. |
| Twitter / X | [@ndif_team](https://x.com/ndif_team) | Release announcements, research highlights. |
| Paper | [arXiv:2407.14561](https://arxiv.org/abs/2407.14561) | Cite this if you use nnsight in research. The canonical academic description of nnsight + NDIF. |

## NDIF infrastructure

| Resource | URL | When to go there |
|----------|-----|------------------|
| NDIF login | [https://login.ndif.us](https://login.ndif.us) | Get your NDIF API key (paste into `CONFIG.set_default_api_key(...)`). |
| NDIF API | [https://api.ndif.us](https://api.ndif.us) | Default value of `CONFIG.API.HOST`. Status, env, job submission, results. |
| NDIF status page | [https://nnsight.net/status](https://nnsight.net/status) | Check which models are currently deployed and running (or call `nnsight.status()`). |

## Tutorials at nnsight.net

The documentation site hosts a curated set of interpretability tutorials built on nnsight. Direct deep links to the most-referenced ones:

| Tutorial | URL | What it shows |
|----------|-----|----------------|
| Walkthrough | [nnsight.net/notebooks/tutorials/walkthrough/](https://nnsight.net/notebooks/tutorials/walkthrough/) | End-to-end intro — tracing, generation, batching, gradients, sessions, remote. |
| Logit lens | [nnsight.net/notebooks/tutorials/logit_lens/](https://nnsight.net/notebooks/tutorials/logit_lens/) | Decoding hidden states from intermediate layers via the unembedding. |
| Activation patching | [nnsight.net/notebooks/tutorials/activation_patching/](https://nnsight.net/notebooks/tutorials/activation_patching/) | Causal tracing — copy a clean run's activation into a corrupted run. |
| Attribution patching | [nnsight.net/notebooks/tutorials/attribution_patching/](https://nnsight.net/notebooks/tutorials/attribution_patching/) | Linearized activation patching using gradients (`tensor.backward()`). |
| Dictionary learning / SAEs | [nnsight.net/notebooks/tutorials/dictionary_learning/](https://nnsight.net/notebooks/tutorials/dictionary_learning/) | Training and applying sparse autoencoders inside a trace; using `hook=True` on aux modules. |
| Circuit finding | [nnsight.net/notebooks/tutorials/circuit_finding/](https://nnsight.net/notebooks/tutorials/circuit_finding/) | Identifying minimal subgraphs of a model that implement a specific behavior. |

If a specific link 404s, fall back to the canonical Tutorials section at [nnsight.net](https://nnsight.net) — paths can be reorganized between releases.

## Internal docs (in this repo)

| Doc | Path | When to read |
|-----|------|--------------|
| `README.md` | [`../../README.md`](../../README.md) | High-level intro, install, quick start. |
| `STYLE.md` | [`../../STYLE.md`](../../STYLE.md) | Code/doc style conventions for this repo. |

## Related projects

| Project | URL | What it is |
|---------|-----|------------|
| NDIF skills marketplace | [https://github.com/ndif-team/skills](https://github.com/ndif-team/skills) | Claude Code / Codex plugin with nnsight skills. |
| nnterp | [https://github.com/ndif-team/nnterp](https://github.com/ndif-team/nnterp) | nnsight-based library standardizing transformer interfaces across model families. |
| nnsight-vllm-demos | [https://github.com/ndif-team/nnsight-vllm-demos](https://github.com/ndif-team/nnsight-vllm-demos) | Demo apps including async chat with SAE-based steering. |
| DeepWiki page | [https://deepwiki.com/ndif-team/nnsight](https://deepwiki.com/ndif-team/nnsight) | Auto-generated wiki explorer for the nnsight codebase. |
