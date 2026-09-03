---
title: Source Tracing
one_liner: Reach intermediate operations inside a module's forward via .source.<callable>_<n>, and drill into called functions.
tags: [usage, source, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/skip.md, docs/usage/cache.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/envoy.py]
---

# Source Tracing

## What this is for

`module.source` exposes every call site inside a module's `forward` as an
addressable operation. nnsight rewrites the module's `forward` AST so each call
`fn(*args, **kwargs)` is bracketed by the interleaver: you can read/replace each
operation's `.input` / `.output`, or `.skip` it — the same handles a module has,
one level finer.

Use it when the value you need lives *between* two operations of a forward (e.g. an
activation inside an MLP, or attention scores) and there is no submodule to attach
to.

## When to use / when not to use

- Use when the activation you need is computed mid-forward and isn't a child module.
- Use to read or replace a single operation's output.
- Skip when a child module already exposes the value — `model...mlp.output` is
  cheaper than `.source` (no AST rewrite).
- Source-instrumentation is installed lazily on first `.source` access and stays on
  the module for the life of the process. Outside a trace each operation calls
  straight through, which is cheap but not free: sourcing every block, attention and
  MLP of GPT-2 costs about 6% on a forward pass with no trace running.

## Discovering operations

`print(module.source)` renders the forward with each operation labelled at its
call site (works outside a trace):

```python
from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", dispatch=True)

print(model.transformer.h[0].mlp.source)
```

```
                     * def forward(self, hidden_states: ...) -> torch.FloatTensor:
 self_c_fc_0     ->  0     hidden_states = self.c_fc(hidden_states)
 hidden_states_0 ->  +     ...
 self_act_0      ->  1     hidden_states = self.act(hidden_states)
 hidden_states_1 ->  +     ...
 self_c_proj_0   ->  2     hidden_states = self.c_proj(hidden_states)
 hidden_states_2 ->  +     ...
 self_dropout_0  ->  3     hidden_states = self.dropout(hidden_states)
 hidden_states_3 ->  +     ...
                     4     return hidden_states
```

Each line carries two operations: the call, and the assignment that binds its
result (`hidden_states_n`). `print(module.source.self_act_0)` zooms in on
one call site:

```
model.transformer.h.0.mlp.source.self_act_0:

    def forward(self, hidden_states: ...) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
    --> hidden_states = self.act(hidden_states) <--
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
```

Operation names come from the library's own source, so they move when the library
does: every label in this page was read off `transformers` 5.15, and a version that
renames an internal variable renames its operation. Print the source for the version
you are running rather than copying a label from anywhere, including here — the
listing is the only thing that is right by construction, and an `AttributeError` on
a wrong name lists the available ones.

## Operation naming

Names are `<dotted_callee>_<occurrence>`, using the **whole** attribute chain
joined with `_`:

- `self.c_fc(...)` → `self_c_fc_0`
- `self.act(...)` → `self_act_0`
- `torch.relu(...)` → `torch_relu_0`

The occurrence counter is per callable and runs in **execution order** — nested
calls run inner-first, so `f(f(x))` gives the inner call `f_0` and the outer `f_1`.
A second `relu(...)` on another line is `torch_relu_1`.

Every assignment is an operation too, named `<target>_<occurrence>` — the
n-th binding of that name in the forward:

- `scores = q @ k.transpose(-1, -2)` → `scores_0` (the product; the call is
  `k_transpose_0`)
- `state = state * decay + update` inside a loop → `state_1` (`_0` is the
  initialisation before the loop), fired once per iteration
- `a, b = x * 2, x * 3` → `a_0`, `b_0`
- `out[:, i] = v` → `out_n`, whose `.output` is `v` (the value stored, not
  the whole tensor); `self.buf = v` → `self_buf_n`

An assignment's `.output` is the assigned value; assigning to it rebinds the name
for the rest of the forward. `.input` is the same value, `.skip(v)` binds `v`
instead, and `.source` raises — there is no callee to drill into. Calls and
assignments share one counter per name, so where a forward binds a name and then
calls it, the binding comes first: in GPT-2's attention,
`attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)` is
`attention_interface_0` and the call `attention_interface(...)` is
`attention_interface_1`. Chained (`a = b = v`) and augmented (`x += v`)
assignments are not operations, and only a **tuple literal** on the right-hand side
names its targets — `g, h = torch.chunk(x, 2)` gives you `torch_chunk_0` and
nothing called `g_0`.

Because the value is whatever the right-hand side evaluated to, an assignment's
`.output` need not be a tensor. `attention_interface_0` above holds a *function*;
`hidden_shape_0` holds a shape tuple, `is_cross_attention_0` a `bool`, `range_0` a
`range`. Nothing raises when you save one — you just get the object, so check what
you got:

```python
with model.trace("Hello"):
    fn = model.transformer.h[0].attn.source.attention_interface_0.output.save()
assert callable(fn)                          # the implementation, not a tensor
assert fn.__name__ == "eager_attention_forward"
```

## Canonical pattern

```python
with model.trace("Hello"):
    # read an operation's output
    act = model.transformer.h[0].mlp.source.self_act_0.output.save()

    # read its inputs as (args, kwargs)
    args, kwargs = model.transformer.h[0].mlp.source.self_c_proj_0.inputs

    # replace an operation's output for the rest of the forward
    model.transformer.h[0].mlp.source.self_c_proj_0.output[:] = 0
```

`.input` is the first argument; `.inputs` is the full `(args, kwargs)`. Assigning to
any of them replaces the value downstream. Operations must be requested in
execution order within a forward (see Gotchas).

## Recursive source — drill into a called function

An operation whose target is a plain Python function can be re-traced. Chain
`.source` again to expose *its* operations:

```python
with model.trace("The Eiffel Tower is in"):
    attn = model.transformer.h[0].attn
    scores = (
        attn.source.attention_interface_1
            .source.attn_output_transpose_0
            .output.save()
    )
```

**Recursive `.source` only works inside a trace.** The called function is resolved
from the live value flowing through the call at run time (call targets are often
local variables), so `some_op.source` outside a trace raises `SourceNotAvailable`.

Drill only when you have to. Attention probabilities, for instance, are reachable
three ways under `attn_implementation="eager"` — `attention_interface_1.output[1]`,
and inside the interface either `nn_functional_softmax_0.output` or
`attn_weights_2.output` — and the three are the same tensor. The first needs no
drill, so it avoids the ordering constraint above and survives a rename of the
interface's local variables; the other two survive a change to what the interface
*returns*. Under `sdpa` the fused kernel never builds the matrix and
`attention_interface_1.output[1]` is `None`, so load the model with
`attn_implementation="eager"` when you want it.

## Iteration support

Source operations participate in `tracer.iter[...]`. Operation iteration counts
**invocations**, not forward passes:

```python
# nested source across generation steps
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    saved = nnsight.save([])
    for _ in tracer.iter[:3]:
        saved.append(
            model.transformer.h[0].attn
                 .source.attention_interface_1
                 .source.attn_output_transpose_0.output
        )
# len(saved) == 3; step 0 sees the full prompt, later (KV-cached) steps one token.
```

An op that fires once per forward is indexed per generation step; an op that loops
within one forward (e.g. an MoE expert loop) is indexed per fire.

## Values that aren't calls: a loop's running state

A recurrent kernel carries its state through a Python loop as a product, never as
a call's return value, so only its assignment names it. The assignment is one
location fired per iteration; `tracer.iter` selects the fire:

```python
# Qwen3.5-MoE linear attention: torch_chunk_gated_delta_rule advances the
# recurrent state once per 64-token chunk, `last_recurrent_state = (...)`.
with model.trace(prompt) as tracer:
    layer = model.model.language_model.layers[0]      # this family nests a language_model
    kernel = layer.linear_attn.source.torch_chunk_gated_delta_rule_0.source
    states = nnsight.save([])
    for _ in tracer.iter[:3]:                                   # chunks 0, 1, 2
        states.append(kernel.last_recurrent_state_1.output)  # _0 is the init
    for _ in tracer.iter[1]:
        kernel.last_recurrent_state_1.output[:] = 0        # patch after chunk 1
```

## Decorated forwards and dispatchers

A decorated `forward` is instrumented through its decorators. A wrapper that
calls the function it closes over — `functools.wraps` or not — is peeled, the
function is instrumented, and the wrapper is rebuilt around it so its behaviour
still runs (`@torch.no_grad()` still disables grad). A wrapper that *doesn't* call
it — a dispatcher that hands the function to a lookup and calls the result — is
instrumented as it is, closure intact, so `print(module.source)` shows the
dispatch and the call that actually runs is what you drill into:

```python
# transformers' experts run a fused kernel by default; the eager loop the class
# defines never executes. experts.source shows the dispatch, not the loop:
print(model.model.layers[0].mlp.experts.source)
#  experts_interface_get_interface_0 -> 1  experts_forward = experts_interface.get_interface(...)
#  experts_forward_0            -> +  ...
#  experts_forward_1                 -> 2  return experts_forward(self, *args, **kwargs)

with model.trace(prompt):
    impl = model.model.layers[0].mlp.experts.source.experts_forward_1.source
    # grouped_mm_experts_forward's ops under the default implementation, the
    # eager loop's under experts_implementation="eager"
```

## Operations that never run

`print(module.source)` and `.names` decompose the whole `forward`, branches
included — so they name operations under an `if` this model's config makes false.
Asking for one parks a request at a location the model never reaches, and the
trace fails with `OutOfOrderError`: a message about being late, for what is really
a dead label.

GPT-2's attention lists 50 operations and runs 28. The other 22 are the
cross-attention and cache-hit paths a plain GPT-2 forward never enters, and their
labels sit right beside the live ones:

```python
# transpose_0 is the cross-attention key transpose. It never runs.
with model.trace(prompt):
    key = model.transformer.h[0].attn.source.transpose_0.output.save()
# OutOfOrderError: '...attn.source.transpose_0.output.i0' was requested but
#                  the model already ran past it

# transpose_2 is the key transpose GPT-2 actually performs.
with model.trace(prompt):
    key = model.transformer.h[0].attn.source.transpose_2.output.save()
assert key.shape[1] == 12 and key.shape[-1] == 64   # [batch, heads, seq, head_dim]
```

When an operation you can see in the listing reports as run past, read the lines
around it before suspecting your request order. One under a false condition —
`is_cross_attention`, an `is_updated` cache hit, `use_parallel_residual` — is
dead, and the live twin is usually the next occurrence of the same name.
Requesting one operation per trace tells you which labels answer.

## Gotchas

- **Recursive `.source` requires a live trace** — outside one it raises
  ``SourceNotAvailable("recursive `.source` is only available inside a trace")``.
- **Drilling into a submodule call is refused.** If an operation calls a
  `torch.nn.Module`, `op.source` raises ``SourceNotAvailable("'self_c_fc_0' calls a
  submodule; call `.source` on that submodule directly instead of drilling into the
  call")``. Access the submodule's own `.source` instead.
- **Builtins / C functions have no recoverable source** — drilling into e.g.
  `torch.relu` raises `SourceNotAvailable`.
- **A dispatching wrapper's ops are the dispatch, not the body.** When a
  decorator chooses an implementation at run time (transformers'
  `experts_implementation`), the class's own `forward` may never run; its calls
  are reached through the dispatch op's `.source` (see
  [above](#decorated-forwards-and-dispatchers)).
- **Requesting an operation out of execution order deadlocks → `OutOfOrderError`.**
  Reading `self_fc2_0.output` then `self_fc1_0.output` (fc1 runs first) is late.
  The same error means something else entirely when the operation is on a branch
  that never runs (see [above](#operations-that-never-run)).
- **A drill is itself an ordered request, one step earlier than the operation's
  own output.** `op.source` is served just before the call runs, so it has to be
  asked for before `op.output`:

    ```python
    with model.trace(prompt):
        inner = attn.source.attention_interface_1.source     # drill first
        probs = inner.attn_weights_2.output.save()
        both  = attn.source.attention_interface_1.output.save()   # then the output
    ```

    The other order raises `OutOfOrderError` on `...attention_interface_1.fn`.
- **The *first* `.source` access on a module has to happen before that module's
  forward runs.** Instrumenting an operation rewrites the module's forward, so if
  the trace body already read something, the model is mid-pass by the time the
  request lands: the instrumentation misses this pass and you get
  `OutOfOrderError`. It is **per module** — warming `h[5].attn` does nothing for
  `h[7].attn`, so a layer sweep hits it once per layer. A bare attribute access
  outside the trace is the whole fix, and costs no forward pass:

    ```python
    _ = model.transformer.h[5].attn.source          # instrument now

    with model.trace(prompt):
        qkv     = model.transformer.h[5].attn.c_attn.output.save()
        pattern = model.transformer.h[5].attn.source.attention_interface_1.output[1].save()
    assert tuple(qkv.shape)[-1] == 2304 and pattern.shape[1] == 12
    ```

    Warming the module also covers drills below it, so `attention_interface_1.source`
    later in the block needs nothing extra. Putting the `.source` request first
    inside the block works too, but only when nothing you need runs earlier in the
    forward; the warm-up has no such constraint.
- **Skipping a whole module drops its source ops.** A skipped module's body never
  runs, so reading `skipped_module.source.<op>.output` is out of order. See
  [skip.md](skip.md).
- **Unknown operation names raise `AttributeError` listing the available names** —
  handy for fixing a mistyped or wrong-occurrence label.

## When `.source` isn't available

`SourceNotAvailable` means there is no Python source to instrument: a builtin or
C function (`torch.relu`), a function compiled from a string, or a call into a
`torch.nn.Module` (use that submodule's own `.source`). Decorators and closures
are not a reason — a wrapped `forward` is peeled or instrumented as it is, and a
`forward` using `super()` keeps its `__class__` cell.

A function that refers to its own name in its body — which every
`torch.nn.functional` entry point does, to pass itself to the dispatcher — cannot
be drilled into: `nn_functional_softmax_0.source` raises `KeyError: 'softmax'`.
Its `.output` is unaffected, and for `F.softmax` that is the value you wanted
anyway.

A value can also be out of reach because it is never computed in Python: with a
fused attention kernel (`attn_implementation="sdpa"`, flash) there is no
attention pattern to read, and a Triton linear-attention kernel (`fla`) has no
per-step state. Choose the eager implementation when you need those.

Where an operation is unavailable, target a real submodule instead, which is
often the value you wanted anyway:

```python
attn = model.model.layers[0].self_attn      # Qwen2.5-0.5B: 14 heads, 2 KV heads, hidden 896

attn.q_proj.output        # (1, 7, 896)  queries, flat — heads not yet split out
attn.v_proj.output        # (1, 7, 128)  values, flat KV-head space under GQA
attn.o_proj.input         # (1, 7, 896)  attention output, already flattened
model.model.layers[0].mlp.act_fn.output   # the MLP's activation
```

A projection's own output is always flat `(batch, seq, features)`: the head split
happens in the `forward`, between the projection and the attention call, so it is
an operation and not a module. When you need `[batch, heads, seq, head_dim]`, that
is `attn.source.query_states_0` / `key_states_0` / `value_states_0` on a
Llama-family attention — `(1, 14, 7, 64)` and `(1, 2, 7, 64)` for the model
above.

`print(module.source)` lists the op names — the quickest way to see what a
version of a model exposes, and whether a wrapper dispatches elsewhere.

## Related

- [access-and-modify.md](access-and-modify.md) — module-level `.output` / `.input`.
- [skip.md](skip.md) — `.skip()` at the module and operation level.
- [iter-all-next.md](iter-all-next.md) — iteration semantics.
