---
title: Source Tracing
one_liner: Reach intermediate operations inside a module's forward via .source.<callable>_<n>, and drill into called functions.
tags: [usage, source, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/skip.md, docs/usage/cache.md, docs/usage/iteration.md, docs/patterns/attention-patterns.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/envoy.py]
---

# Source Tracing

`module.source` exposes every call site and assignment inside a module's
`forward` as an addressable operation with the same handles a module has —
`.input` / `.inputs` / `.output`, `.skip` — one level finer. Use it when the value
you need lives *between* two operations of a forward (an activation inside an MLP,
attention scores before softmax) and no child module exposes it; when one does,
`module.output` is cheaper. Instrumentation is installed on first `.source` access
and stays on the module for the life of the process; outside a trace each
operation calls straight through, at a small cost.

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
result (`hidden_states_n`). `print(module.source.self_act_0)` shows one call site
in context.

Operation names come from the library's own source, so they move when the library
does. Print the listing for the version you are running rather than copying a
label from anywhere, including this page; an `AttributeError` on a wrong name
lists the available ones.

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

An assignment's `.output` need not be a tensor: `attention_interface_0` above
holds a *function*, `hidden_shape_0` a shape tuple, `is_cross_attention_0` a
`bool`. Nothing raises when you save one, so check what you got.

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
from the live value flowing through the call (call targets are often local
variables), so outside a trace it raises
``SourceNotAvailable("recursive `.source` is only available inside a trace")``.

Drill only when you have to: each drill adds an ordering constraint (see Gotchas)
and depends on the inner function's variable names. For attention probabilities,
`attention_interface_1.output[1]` under `attn_implementation="eager"` needs no
drill; see [attention-patterns](../patterns/attention-patterns.md).

## Iteration support

Source operations participate in `tracer.iter[...]`, counting **invocations**, not
forward passes: an op that fires once per forward is indexed per generation step;
one that loops within a forward (an MoE expert loop) is indexed per fire. Read ops
before the module output that contains them, and warm `.source` before the trace
(below).

## Values that aren't calls: a loop's running state

A recurrent kernel carries its state through a Python loop as a product, never a
call's return value, so only its assignment names it. That assignment is one
location fired per loop iteration, and `tracer.iter` selects the fire:

```python
with model.trace(prompt) as tracer:
    kernel = linear_attn.source.torch_chunk_gated_delta_rule_0.source
    states = nnsight.save([])
    for _ in tracer.iter[:3]:                                # chunks 0, 1, 2
        states.append(kernel.last_recurrent_state_1.output)  # _0 is the init
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

`print(module.source)` decomposes the whole `forward`, branches included, so it
names operations under an `if` this model's config makes false. Asking for one
fails with `OutOfOrderError` — a message about being late, for what is really a
dead label. GPT-2's attention lists 50 operations and runs 28; the rest are the
cross-attention and cache-hit paths:

```python
# transpose_0 is the cross-attention key transpose. It never runs.
with model.trace(prompt):
    key = model.transformer.h[0].attn.source.transpose_0.output.save()
# OutOfOrderError: '...attn.source.transpose_0.output.i0' was requested but
#                  the model already ran past it

# transpose_2 is the key transpose GPT-2 actually performs.
with model.trace(prompt):
    key = model.transformer.h[0].attn.source.transpose_2.output.save()
```

When a label you can see reports as run past, read the lines around it before
suspecting your request order. One under a false condition (`is_cross_attention`,
an `is_updated` cache hit, `use_parallel_residual`) is dead, and the live twin is
usually the next occurrence of the same name.

## Gotchas

- **Requesting an operation out of execution order deadlocks → `OutOfOrderError`.**
  Reading `self_fc2_0.output` then `self_fc1_0.output` (fc1 runs first) is late.
  On a branch that never runs, the same error means a dead label (above).
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
- **Touch a module's `.source` before the trace.** The first access instruments
  the forward; if the module is already mid-pass when it lands, this pass is
  missed and you get `OutOfOrderError` — or, inside a `tracer.iter` loop, the read
  silently binds to the op's next call and the list comes up one short
  ([iteration pitfalls](iteration.md#pitfalls)). It is **per module**, and also
  covers drills below it. A bare attribute access outside the trace is the fix:

    ```python
    _ = model.transformer.h[5].attn.source          # instrument now

    with model.trace(prompt):
        qkv     = model.transformer.h[5].attn.c_attn.output.save()
        pattern = model.transformer.h[5].attn.source.attention_interface_1.output[1].save()
    ```
- **Skipping a whole module drops its source ops.** A skipped module's body never
  runs, so reading `skipped_module.source.<op>.output` is out of order. See
  [skip.md](skip.md).
- **Unknown operation names raise `AttributeError` listing the available names** —
  handy for fixing a mistyped or wrong-occurrence label.

## When `.source` isn't available

`SourceNotAvailable` means there is no Python source to instrument:

- a builtin or C function (`torch.relu`);
- a function compiled from a string;
- a call into a `torch.nn.Module`: ``SourceNotAvailable("'self_c_fc_0' calls a
  submodule; call `.source` on that submodule directly instead of drilling into
  the call")`` — use that submodule's own `.source`;
- a recursive drill outside a trace (above).

Decorators and closures are not a reason, and a `forward` using `super()` keeps
its `__class__` cell. A function that refers to its own name in its body — every
`torch.nn.functional` entry point does — cannot be drilled into:
`nn_functional_softmax_0.source` raises `KeyError: 'softmax'`; its `.output` is
unaffected.

A value can also be out of reach because it is never computed in Python: a fused
attention kernel (`sdpa`, flash) builds no attention pattern, and a Triton
linear-attention kernel (`fla`) has no per-step state. Choose the eager
implementation when you need those. Otherwise target a real submodule: a
projection's output (`q_proj`, `v_proj`) is flat `(batch, seq, features)` because
the head split happens in the `forward`, so per-head `[batch, heads, seq,
head_dim]` tensors are operations (`query_states_0` / `key_states_0` /
`value_states_0` on a Llama-family attention); see
[per-head-attention](../patterns/per-head-attention.md).
