# Questions

Ambiguities for the human to resolve, raised while porting the `docs/` from old
nnsight (`/home/localjadenfk/wd/nnsight`) to this pipeline-based rewrite. Any agent
(or the orchestrator) appends questions here; answers go inline under each.

Format:
```
## <area>: <short question>
<context — what's ambiguous, what the options are, what default was assumed>
**Answer:** (human fills in)
```

---

## developing part B: dropped agent-evals.md (subsystem removed)
The OLD `docs/developing/agent-evals.md` documented `tests/agent-evals/` (a
Claude/GPT harness measuring whether agents can write nnsight code from the docs).
That directory **does not exist** in this repo (no `tests/agent-evals/`, no
`agent.py`/`eval.py`/`tasks/`, and the `anthropic`/`openai` SDK deps are gone).
**Default assumed: drop the page entirely** — nothing to document. If the eval
harness is re-added later, restore it from the OLD doc. The top-level index/router
(orchestrator-owned) should drop any link to `agent-evals.md`.
**Answer:** leave this for now ell come back

## developing part B: contributing.md sourced from STYLE.md (no CONTRIBUTING.md)
The OLD `contributing.md` pointed at root `CONTRIBUTING.md`, `CLAUDE.md`, and
`NNsight.md` — **none of which exist** in this repo. This repo has `STYLE.md` (the
canonical house style) instead. **Default assumed: rewrote contributing.md around
`STYLE.md` plus the real branch (`main`) and commit conventions (`area: summary`
subject + `Co-Authored-By: Claude ...` trailer, both observed in git log).** The
Discord/forum/nnsight.net links were kept (external, still plausibly valid) — verify
they're current.
**Answer:**

## developing part B: vllm-integration.md — OLD referenced files are gone
The OLD vllm doc leaned on `src/nnsight/modeling/vllm/README.md` (+ DISCUSSION.md,
IDEAS.md) as the "canonical narrative" and on `sampling.py`/`executors/ray_workaround.py`.
**None exist now.** Interventions ride stock `SamplingParams.extra_args["nnsight_mediator"]`;
Ray uses vLLM's stock backend; async is `tracer.py`/`VLLMTracer`; a new `serve/`
package adds an HTTP path. **Default assumed: documented the code as-is and dropped
all links to the missing files.** The module docstring in `vllm.py` is now the
narrative reference.
**Answer:**

## gotchas: vLLM pipeline-parallelism claim not reproduced
OLD `gotchas/integrations.md` claimed vLLM PP is unsupported and
`pipeline_parallel_size` is silently forced to 1 while TP/DP are supported
(citing `vllm/vllm.py:139`). In the rewrite the only forcing I can see
(`vllm/vllm.py:164`) sets both `tensor_parallel_size=1` and
`pipeline_parallel_size=1` **for the meta-tree build only**, not the real engine,
and the intervention surface changed (`model.logits`/`model.samples`, sampling
kwargs on `trace`/`invoke`, `mode="sync"|"async"`). I can't run vLLM here (no GPU),
so I rewrote the vLLM section conservatively around the verified-by-source usage and
pointed engine/parallelism details to `docs/models/vllm.md` rather than restate the
PP/TP support claim. Default assumed: **omit the PP-unsupported/TP-supported claim
from the gotcha; leave parallelism to the vLLM model doc.**
**Answer:**

## gotchas: iteration "generate sets default_all so trailing code runs" is false now
OLD `gotchas/iteration.md` said `generate(..., max_new_tokens=N)` sets
`interleaver.default_all = N`, so a plain `tracer.iter[:]` inside `generate`
terminates and trailing code (saving `generator.output`/`tracer.result`) runs. In
the rewrite there is **no `default_all`** — unbounded `tracer.iter[:]`/`all()`
always leaves a dangling final request, the interleaver throws `OutOfOrderError`
into the worker (caught, warned) and unwinds it, so **any** code after the loop is
skipped (verified: `ids = tracer.result.save()` after `for _ in tracer.iter[:]` is
never defined). Rewrote the doc to say: bound the loop, or move trailing code into a
separate empty `tracer.invoke()`. Also `.next()`/`tracer.next()` don't exist.
Default assumed: **document unbounded iter as always dropping trailing code.**
**Answer:**
