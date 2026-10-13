---
title: Block Capture Errors
one_liner: "Errors raised as a traced `with` block is entered: WithBlockNotFoundError (no recoverable `with`), a body starting with `try:`, a body on the `with` line, and `return` inside the body."
tags: [error, tracing, ast]
related: [docs/errors/index.md, docs/errors/debug-mode.md, docs/concepts/deferred-execution.md]
sources: [src/nnsight/tracing/tracer.py]
---

# Block Capture Errors

nnsight runs a traced block's body itself, so it has to find the `with` statement in
the source, compile its body, and stop the interpreter from running that body inline.
Each error here is one of those steps refusing, before anything runs.

## `WithBlockNotFoundError`

```
WithBlockNotFoundError: nnsight has no source for <stdin>, so it cannot read the
body of the `with` block at line 3. …
WithBlockNotFoundError: nnsight found no `with` statement at run.py:12, the line
the trace was entered from, which reads `t.__enter__()`. …
WithBlockNotFoundError: nnsight found no `with` statement at <ipython-input-3>:4,
the line the trace was entered from. …
```

nnsight reads the source of the frame that entered the tracer and looks for a
`with` at that line. Each message is one way that fails:

- **No source at all** — a script fed on stdin (`python < script.py`,
  `cat script.py | python`). Run the file by name; `python -c "..."` is fine.
- **No `with` at that line** — the tracer was entered by hand
  (`model.trace(x).__enter__()`, or stored and entered later).
- **Source doesn't line up with the running code** — code `exec`/`compile`-d from a
  string not registered in `linecache`, or a file edited after nnsight first read
  it (each file's source is read once).

```python
# WRONG
t = model.trace("Hello")
t.__enter__()                       # WithBlockNotFoundError

# FIXED
with model.trace("Hello"):
    out = model.lm_head.output.save()
```

For programmatic interventions, put the trace in a real function in a real file and
call it rather than generating source at runtime. If an ordinary `with` block in a
saved file reproduces this from a clean state, it is a bug — open an issue.

## A trace body that starts with `try:`

```
ValueError: A traced `with` block cannot start with `try:`; nnsight intercepts the
body at its first line, and a `try` there is the one statement Python gives it no
way back out of. Put any statement above the `try`, or move the `try` outside the
block.
```

nnsight stops the interpreter from running the body inline by raising at the body's
first line and catching that raise in the `with`. CPython puts the `try` keyword's line on an instruction that no
exception-table entry covers, so a raise delivered there unwinds the frame
outright instead of reaching the `with`. The shape is refused rather than
silently losing the whole block.

Either statement below fixes it:

```python
with model.trace(prompt):
    hidden = model.transformer.h[0].output      # any statement above the try
    try:
        head = hidden[0, -1]
    except IndexError:
        head = None
    kept = nnsight.save(head)
```

```python
try:                                            # or wrap the whole with
    with model.trace(prompt):
        hidden = model.transformer.h[0].output.save()
except Exception as error:
    print(error)
```

Only the *first* statement is the cue — a `try` anywhere else in the body is
ordinary Python and catches what it is written to catch.

## A body on the `with` line

```
ValueError: The body of a traced `with` must start on its own line; nnsight runs
the body itself, and can only intercept it at the start of a line.
```

Same mechanism, same fix: give the body its own indented line.

```python
with model.trace(prompt): out = model.output.logits.save()      # refused

with model.trace(prompt):
    out = model.output.logits.save()                            # fine
```

## `return` inside a trace body

The body is compiled on its own, outside any function, so a `return` in it is a
syntax error at capture time:

```
SyntaxError: 'return' outside function
```

Return the saved value *after* the block:

```python
def hidden_state(prompt):
    with model.trace(prompt):
        hidden = model.transformer.h[0].output.save()
    return hidden
```

## Related

- [symptom-index.md](symptom-index.md) — failures that raise nothing.
- [docs/concepts/deferred-execution.md](../concepts/deferred-execution.md) — why nnsight needs the block's source in the first place.
- [debug-mode.md](debug-mode.md) — traceback handling.
