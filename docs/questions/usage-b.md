# Usage Docs (Part B) Questions

## docs/usage/index.md
1. The index links to docs that other agents are writing (trace, generate, invoke-and-batching, save, access-and-modify, scan, edit, session, conditionals-and-loops, skip, barrier, stop-and-early-exit, rename-modules). Filenames inferred from the user's task description — confirm the exact filenames expected so dead links don't appear.
2. Should the index group by category (as I did) or be a flat alphabetical list? I went with category headings since this is the routing surface for AI agents.

## docs/usage/source.md
1. The "first-time `.source` access mid-iter-loop misses one step" limitation has a noted future fix ("seed op-path trackers from the parent module's tracker at `Envoy.source` access time"). If that lands before docs ship, this gotcha goes away — flag for re-check before release.
2. `SourceAccessor.__iter__` deduplicates by line number, so if two operations sit on the same line only the first is yielded by iteration but both are still callable by name. I didn't surface this — confirm that's the right call or whether it deserves a one-line note.
3. Operation naming uses `_<index>` to disambiguate repeated calls. I gave the rule but didn't enumerate every edge case (e.g. attribute chains like `self.a.b.c(...)` flatten to `self_a_b_c_0`). Worth a more exhaustive table?

## docs/usage/cache.md
1. Cache hooks fire after intervention hooks because `mediator_idx = float('inf')`. If a future change introduces a hook category that fires *after* cache hooks, the "post-intervention values" promise becomes leaky. Worth an explicit version note pinning this to current behavior, or trust the source ref?
2. The `Cache.Entry` becoming `list[Entry]` on repeated hits — I documented this but the user-facing shape is unintuitive (sometimes `cache[path].output` works, sometimes `cache[path][0].output`). Should I show the list-vs-single dispatch explicitly with a code example?
3. `tracer.cache(modules=[envoy_or_string])` accepts both Envoy objects and string paths. The implementation walks `model.modules()` once for string paths to resolve them. Worth flagging that mixing the two kinds in one call is fine, or is that obvious?

## docs/usage/iter.md
1. There are two unbounded-iter footguns conflated in CLAUDE.md ("`tracer.iter[:]`" and "`tracer.all()`"). I split them across iter.md and all-and-next.md. Confirm that's the right split rather than centralizing in one of them.
2. The `iteration_tracker` is per-mediator. Cross-invoke iter loops would each have their own tracker. Is there a real use case for cross-invoke iter (e.g. running different per-step interventions on different invokes), and is it worth a recipe?

## docs/usage/all-and-next.md
1. `tracer.next()` returns `self` (the tracer) but `module.next()` returns `self` (the Envoy). Both flow into the same `mediator.iteration` bump but the chaining ergonomics differ. Worth showing chained `model.transformer.h[-1].next().next().output`?
2. `module.next()` is deprecated. Should this doc still cover it, or only mention it under "deprecated" once?

## docs/usage/backward-and-grad.md
1. The patch is on `torch.Tensor.backward` (module-level) at import time. If a user does `from torch import Tensor` *before* `import nnsight`, do they get the patched or unpatched version? I believe both names point to the same class so it works either way, but worth a quick experimental confirmation.
2. `BackwardsTracer` re-uses the `Invoker` capture path. Errors that escape the backward block are caught by the invoker's compiled try/catch. Worth documenting the user-visible exception flow (e.g. `IndexError` inside backward shows up where in the traceback)?
3. The CLAUDE.md "tensor IDs in error messages" note (`139820463417744.grad`) — confirm this is still accurate; with the `id(tensor)` keying I assume yes, but worth a smoke test before final.

## docs/usage/extending.md
1. The SAE worked example elides the `_last_output` capture (used to rebuild the residual tuple after `transform`). It needs a real implementation pattern (e.g. an `eproperty` with a sibling `preprocess` that stashes the original onto `self`). Should I expand this into a complete runnable example, or is it clearer as illustrative pseudocode?
2. `eproperty.provide(obj, value)` is the external-push API for runtime-injected values. I documented its use (vLLM logits/samples) but didn't show the full integration scaffolding (how the runtime locates `obj`, how `iterate=` interacts with the request key). Worth a separate "writing a runtime adapter" recipe, or out of scope for this doc?
3. Custom hook setup decorators (analogues of `requires_output` for non-PyTorch backends) are alluded to but not shown. Worth a complete example, or does the vLLM source serve as the canonical reference?
4. The `IEnvoy` Protocol declares `path: str` non-Optional but tracer-level eproperties (like `InterleavingTracer.result`) work without a path prefix because `_build_requester` checks for empty/missing path. This inconsistency surfaced in the concepts questions too — should the docs warn extenders that `path = ""` is a valid value?
5. `envoys=` as a class attribute on a `NNsight` subclass: I noted that users can opt out per-instance by passing `envoys=None`, but the `kwargs.setdefault("envoys", type(self).envoys)` semantic means `envoys=None` explicitly will still pick up the class default unless `setdefault` doesn't trigger because the key exists. Need to verify whether passing `envoys=None` actually overrides the class default.
