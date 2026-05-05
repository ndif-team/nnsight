---
title: Session
one_liner: Group multiple traces under one context with `model.session()`; bundle remote requests into a single round-trip.
tags: [usage, session, remote]
related: [docs/usage/trace.md, docs/usage/save.md, docs/usage/conditionals-and-loops.md]
sources: [src/nnsight/__init__.py:116, src/nnsight/intervention/envoy.py:413, src/nnsight/intervention/tracing/base.py:47, src/nnsight/modeling/mixins/remoteable.py:76]
---

# Session

## What this is for

`model.session()` opens a `Tracer` context that **does not run the model itself** — it captures Python code that contains one or more `model.trace(...)` blocks (and arbitrary control flow between them). The captured code is compiled once and executed in order. The two main use cases:

- **Local**: bundle related traces, share variables across them naturally, run loops over many prompts.
- **Remote** (`remote=True`): ship the whole session as a single payload to NDIF — one queue wait, one network round-trip, regardless of how many inner traces there are.

The session itself is a base `Tracer` (no interleaving), so it has no `tracer.invoke`, `tracer.iter`, `tracer.cache`, etc. — those live on the inner `model.trace(...)` contexts.

## When to use / when not to use

- Use locally for clarity when you have several related traces that share variables.
- Use locally to put a Python `for` loop around many traces (one trace per prompt).
- Use **always** for remote workloads with multiple traces — saves queue time and bandwidth.
- For a single forward pass, just use `model.trace(...)` directly.

## Canonical pattern

```python
with model.session() as session:
    with model.trace("Hello"):
        hs = model.transformer.h[0].output.save()

    with model.trace("World"):
        # use hs from the previous trace — it's a real captured tensor
        model.transformer.h[0].output[:] = hs
        out = model.lm_head.output.save()
```

## Looping over prompts

Standard Python `for` works inside a session:

```python
prompts = ["Hello", "World", "Test"]

with model.session():
    results = list().save()
    for prompt in prompts:
        with model.trace(prompt):
            results.append(model.lm_head.output.argmax(dim=-1))
```

See `docs/usage/conditionals-and-loops.md` for `if`/`for` semantics.

## Remote sessions

`RemoteableMixin.session` (`src/nnsight/modeling/mixins/remoteable.py:76`) accepts `remote=True` / `blocking=` / `backend=` kwargs:

```python
with model.session(remote=True):
    # First trace: capture activations
    with model.trace("Megan Rapinoe plays the sport of"):
        hs = model.model.layers[5].output[:, -1, :]   # no .save() needed inside

    with model.trace("Shaquille O'Neal plays the sport of"):
        clean = model.lm_head.output[0][-1].argmax(dim=-1).save()

    with model.trace("Shaquille O'Neal plays the sport of"):
        model.model.layers[5].output[:, -1, :] = hs   # cross-trace reference
        patched = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(clean), "->", model.tokenizer.decode(patched))
```

Properties of remote sessions:

- One job submission, one queue wait, one set of status updates.
- Variables defined in earlier traces are usable in later traces directly — the session frame is shared on the server side.
- Only put `remote=True` on the outer `session()`, not on inner traces.
- `blocking=False` returns immediately; check `tracer.backend.job_status` and call `tracer.backend()` to fetch when ready.

`session(remote=True)` uses `RemoteTracer` (or `RemoteInterleavingTracer` for inner traces, set via `tracer_cls`) to support hybrid remote/local code via `tracer.local()` (`src/nnsight/modeling/mixins/remoteable.py:231`).

## How it works

`Envoy.session(...)` (`src/nnsight/intervention/envoy.py:413`) just constructs a base `Tracer` and attaches `model` to it:

```python
def session(self, *args, tracer_cls: Type[Tracer] = Tracer, **kwargs):
    tracer = tracer_cls(*args, **kwargs)
    setattr(tracer, "model", self)
    return tracer
```

The base `Tracer` (`src/nnsight/intervention/tracing/base.py:47`) captures the with-block body, compiles it into a function, and runs it in the caller's frame via `ExecutionBackend`. Inside that function, every `with model.trace(...):` block is an independent interleaving session — each opens its own worker thread, runs its forward pass, exits, and pushes saved values back into the session frame.

### What `*inputs` does on `model.session(*inputs, ...)`

Positional arguments to `model.session(*inputs, ...)` are forwarded to `Tracer.__init__(*args, ...)` and stored on the tracer as `self.args`. They are then passed to the compiled session function as positional parameters when it runs. **They do NOT propagate into inner `model.trace(...)` blocks** — each inner trace opens its own interleaving session with its own positional arguments.

In practice this is rarely useful — most users define inputs inside the session body. If you do pass `*inputs` here, you're effectively parameterizing the entire compiled session function:

```python
# inputs become positional args of the compiled session function;
# they are NOT auto-forwarded into inner trace(...) calls.
with model.session(some_payload, remote=True) as session:
    with model.trace("Hello"):                   # "Hello" is the trace input here, not some_payload
        out = model.lm_head.output.save()
```

There is also a top-level `nnsight.session(...)` helper (`src/nnsight/__init__.py:116`):

```python
def session(*args, **kwargs):
    return Tracer(*args, **kwargs)
```

This returns a bare `Tracer` not bound to any model. Prefer `model.session(...)` so the model reference is available.

## Sharing values across traces

Locally, intermediate variables that are not `.save()`d but are assigned in the session body are still visible across traces because the session frame is the actual caller frame:

```python
with model.session():
    with model.trace("Hello"):
        hs = model.transformer.h[0].output   # no .save needed for session-scope use

    # Plain Python — runs between traces
    print("captured tensor:", hs.shape)

    with model.trace("World"):
        model.transformer.h[0].output[:] = hs
        out = model.lm_head.output.save()
```

Anything you want **outside the session** still needs `.save()` (or `nnsight.save(...)`).

## Gotchas

- A `session()` on its own does not run the model — you need at least one inner `model.trace(...)`.
- Remote sessions: only the outer `session()` takes `remote=True`. Inner `model.trace(...)` calls inherit the remote backend automatically.
- Session is a `Tracer`, not an `InterleavingTracer` — `tracer.invoke`, `tracer.iter`, `tracer.cache`, `tracer.barrier`, `tracer.result` live on the inner `model.trace` contexts.
- Variables declared inside a session but outside any `.trace()` are normal Python variables — they exist in the session's compiled function frame.
- For values you want **after** the whole session exits, use `.save()` / `nnsight.save(...)` exactly like with `model.trace(...)`.

## Related

- `docs/usage/trace.md`
- `docs/usage/save.md`
- `docs/usage/conditionals-and-loops.md`
- `docs/remote/...` (remote execution specifics)
