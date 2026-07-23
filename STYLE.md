# nnsight style

This is the house style for developing nnsight. It is written for anyone — human
or agent — adding to this codebase. Read the philosophy first: the rules that
follow are consequences of it, and when a rule doesn't cover your case, the
philosophy should tell you what to do.

---

## Philosophy

**nnsight's machinery is subtle, and its subtlety is not visible in the code.**
A forward hook that fires a greenlet switch, a controller that must replace
`forward` before `nn.Module.__call__` binds it, a `cat` chosen over an in-place
write to keep autograd correct — none of that is legible from the statements
themselves. The code says what happens. Everything we write around it exists to
say *why*, and to say it once, in the right place.

That gives the three-layer division this codebase runs on:

- **The module docstring teaches the concept.** What problem this module exists
  to solve, the mechanism it uses, and the constraints that fall out. Read it and
  you can read the module.
- **The docstring on a thing states its contract and the reasoning behind it** —
  not a restatement of the signature.
- **The inline comment explains why *this* implementation**: the language rule,
  the failure mode averted, the alternative rejected.

If a comment paraphrases the line under it, delete it. `VALUE = "value"  # Request
for a value` costs a line and returns nothing. Say the thing the reader can't see:

```python
VALUE = "VALUE"  # read a location:  (Event.VALUE, location)
SWAP = "SWAP"  # replace a location: (Event.SWAP, location, value)
SKIP = "SKIP"  # skip a computation: (Event.SKIP, location, value)
```

**The source is present tense.** It describes what is true now — the invariant,
the constraint, the reason this line is shaped this way. It does not narrate how
it got here. No "this used to", no "fixes issue #479", no TODO/FIXME/HACK. Git
holds history; the source holds the design. When you touch code that exists
because something once broke, write the constraint, not the incident:

```python
# A worker that errors on start (e.g. invoking mid-run) means __exit__
# won't run to clear the flag, so reset it here or it leaks to the next run.
```

**One word, one meaning.** A vocabulary that overloads a term forces every reader
to disambiguate it forever. `swap` names exactly one thing — replacing a value at
a location. Splicing a block's rows back into a batch is `widen`, the true
antonym of `narrow`. When you need a second sense of a word you already use, that
is a signal to find the right word, not to overload the one you have.

**Delete before you add.** The strongest version of a change is usually the one
that removes a concept rather than introducing one. Prefer a plain method to a
custom descriptor; a single dispatch point to a fan-out of `handle_*_event`
functions; one parameter to a near-duplicate function. Before adding an
abstraction, check whether an existing primitive already carries it — and before
adding a layer, check whether the layer below can simply be used correctly.

**Lean on the primitive you already have.** The interleaver exposes exactly one:
a location string and `handle`, which offers a value to interventions and takes
back whatever they wrote. `source` makes the inside of a `forward` addressable
without the interleaver knowing source exists, because it is a client of that one
primitive. The same instinct applies outward: where an upstream library already
knows how to do something — a HuggingFace pipeline's preprocessing — use it
rather than re-deriving it here.

**Layer in one direction.** `tracing` knows nothing of `intervention`;
`intervention` knows nothing of `modeling`. A module that reaches back up its own
stack — or into the root package — has lost the boundary that makes it testable
and readable. `tracing/` is a self-contained "capture a `with` block and run it
through a backend" library, and it stays that way.

**Machinery gets plain names; concepts get the good ones.** `Envoy`, `Mediator`,
`Interleaver` name the ideas users read about and reason with. `_State`,
`Compiled`, `Entry` are unglamorous on purpose — they are plumbing, and a
memorable name for plumbing spends the reader's attention on the wrong thing.

**Write for the reader who is about to change this.** Name the failure mode
averted, the counterfactual, the blast radius. That is the difference between a
comment that survives a refactor and one that gets deleted along with the line.

---

## Structure

**Layering is one-directional: `tracing → intervention → modeling`.** Nothing in
`tracing/` imports from `intervention/` or `modeling/`. Nothing imports the root
`__init__` from inside the package.

**One concept per module, split as soon as it is nameable.** Size is not the
trigger — identity is. `tracing/backend.py` is eleven lines because `Backend` is
one idea and deserves its own file:

```python
class Backend:
    def __call__(self, tracer: Tracer) -> None:
        tracer.execute(tracer.info.code)
```

Modules stay under ~900 lines. A module growing past that usually contains a
second concept that wants its own file.

**Subpackage `__init__.py` files are empty.** Import from the module that defines
the thing (`from nnsight.intervention.envoy import Envoy`). A re-export layer
gives every object two import paths and no benefit.

**The root package is the only public surface, and it stays small.** Model
classes are exposed lazily through a module-level `__getattr__`, so importing
nnsight doesn't drag in transformers or diffusers, and an optional dependency
only errors when its model is actually used. Because a module `__getattr__` is
not consulted by `from nnsight import *`, lazy names are listed in `__all__`.

**Extension points are underscore-prefixed methods with working defaults.** No
ABCs, no abstract mixins — a base implementation that does the sensible thing, or
`raise NotImplementedError` with a message naming what's missing. The docstring
states the base default *and* points at the reference override, so the person
subclassing knows what they're up against:

```python
def _prepare_input(self, *inputs: Any, **kwargs: Any) -> int:
    """Number of batch rows an invoke's input contributes (0 if it has none).

    Base default: any input is a single row. Models that batch (e.g.
    :class:`~nnsight.modeling.transformers.TransformersModel`) override this to
    report the true row count of a prompt / list / tensor.
    """
```

**Inheritance is shallow and load-bearing.** Each class in a chain earns its place
by adding behavior. A class that exists only to be a name is a leaf, not a link —
and its docstring should say so.

**Tests mirror modules.** One test file per source concept, named for it. Behavior
groups into `Test*` classes; the test name carries the assertion, so tests need no
docstrings (`test_assignments_pushed_to_parent`). Prefer inline fakes over mocks,
define small `nn.Module`s locally in the test file, and keep shared model
fixtures — the expensive ones — in `conftest.py`.

---

## Naming

**Full words, short names.** `value`, `location`, `provider`, `group`, `total`,
`full`, `edited`, `served`, `pending`, `worker`. Abbreviate only from the closed
set the codebase already uses: `fn`, `glbls`/`lcls`, `cls`, `args`/`kwargs`, `tb`,
`n`, `i`, `k`/`v`, `pre`/`post`.

**Loop variables are domain nouns, not indices**: `for mediator in ...`,
`for name, child in ...`, `for step in ...`.

**Verbs are methods; nouns are properties.** `.output`, `.input`, `.device`,
`.source`, `.result`, `.batching` are properties. `.trace()`, `.edit()`,
`.skip(replacement)`, `.get(path)`, `.interleave(fn)` are methods.

**Prefix conventions carry meaning:**

| Pattern | Means |
|---|---|
| `_make_x` | returns a new callable/closure |
| `_build_x` | assembles a composite object |
| `_ensure_x` | idempotent get-or-create |
| `_is_x` | boolean predicate |
| `install_x` | module-level; mutates a module in place |

There is no `get_*` prefix. A plain noun is the accessor.

**Underscore everything that isn't user-facing.** The bare surface *is* the public
API — if it has no underscore, someone will call it, and you now own it.

**Classes are `PascalCase`.** Enums are singular (`Event`). No `Mixin` suffix, no
`I` prefix, no `Accessor` suffix — name the thing, not its role in a pattern.
Private classes take a leading underscore (`_State`).

**Exceptions use `-Error` for genuine errors** (`OutOfOrderError`, `RemoteError`).
`-Exception` is reserved for control-flow signals (`EarlyStopException`,
`ExitTracingException`), and their docstrings say outright that they are control
flow and never surface to the user.

**Identifiers injected into someone else's namespace are marked** with an
`nnsight`/`nns` prefix: `__nnsight_op__`, `nns_<name>`, `__intervention_tb__`.

**Module constants are `_UPPER`** unless they are part of the public API. Hoist
magic numbers into named constants rather than inlining them. Sentinels are
module-level `object()` with a `#:` doc comment:

```python
#: Sentinel returned by the skip gate when no skip is pending for a location.
_NO_SKIP = object()
```

---

## Docstrings

**Every core module opens with a docstring that teaches the concept.** This is the
most important convention here. The shape:

1. One-line summary.
2. The problem — why this module exists, in terms a reader who doesn't know the
   codebase can follow.
3. The mechanism, naming its metaphor.
4. A worked example (`.. code-block:: python`, or a `::` literal block).
5. The constraints and gotchas that fall out.

```
"""Interleaving intervention code with a model's forward pass.

nnsight lets you read and edit a model's intermediate values from ordinary
Python written *inside* a ``with model.trace(...):`` block. To make that work,
the intervention code and the model's forward pass have to run in lockstep:
the intervention pauses whenever it asks for a value the model hasn't produced
yet, the model runs until it reaches that value, hands it over, and the
intervention resumes — possibly editing the value on the way back in.
"""
```

Terms of art are introduced in *italics* on first use, then used unglossed forever
after. The load-bearing noun of a module gets **bold** once.

**Class docstrings** open with a one-line summary, then prose on the class's role
and its collaborators. `Attributes:` is reserved for genuinely stateful core
classes, and the entries carry semantics and lifecycle — not types:

```
Attributes:
    worker: The greenlet running :attr:`interventions`, or ``None`` before
        :meth:`start`. Falsy once the worker has finished (see :attr:`alive`).
```

**Method docstrings state the contract and the reasoning.** The summary line sits
on the same line as the opening quotes, imperative, ending in a period. Properties
open with `Whether ...`. For internal machinery, name the parameters inline as
``literals`` and give the return contract in prose — including its edge cases:

```python
def narrow(self, value: Any, group: BatchGroup) -> Any:
    """Slice every batched tensor in ``value`` down to ``group``'s rows.

    A tensor is batched only when its leading dim equals :attr:`total` (the
    combined batch size), so non-batched tensors pass through untouched. Returns
    the whole value when not actually batching or for a groupless (empty) invoke.
    """
```

**The user-facing API carries full `Args:` / `Returns:` / `Examples:`.** This is
the surface people read in generated docs, and prose alone under-serves it.
Everything users touch — `Envoy.trace`, `Envoy.edit`, the `ndif` helpers,
`backward` — documents every parameter, states what it returns, and shows at least
one example. Internal machinery does not: there, the annotation is the type and
the prose is the meaning.

**Types live in annotations, never in the docstring.** Write ``inplace: If
``False`` (default), edit a shallow copy…``, not ``inplace (bool, optional):``.
The `Args:` entry explains the *choice* — what each value means and when you'd
want it — which is the thing the annotation can't say.

**Two example forms, for two jobs.** A standalone `Examples:` section using `>>>`
belongs on the user-facing API, where people read a reference and want a snippet
they can run. An inline `::` literal block belongs mid-docstring, illustrating the
paragraph immediately above it — that's the common form in the machinery, where an
example earns its place by making one sentence concrete. Module docstrings use
`.. code-block:: python`.

**Cross-link aggressively.** `:class:`, `:meth:`, `:attr:`, `:func:`, `:mod:`,
`:data:`, with `~`-shortened targets for anything out-of-module. Double-backtick
every literal, including ``None`` and ``False``. The docstrings form a navigable
graph; that only works if you link.

**Voice:** imperative for mechanics ("Slice every batched tensor…"). Second person
for concepts, addressing the user ("no need to repeat it each time"). First person
plural for the library's own design decisions ("we want to capture it"). Never a
third-person "Returns the…" opener.

**Length is barbelled by design**: long at module level, short at method level. A
one-liner when the summary is the whole contract — a semicolon is fine to join
summary and return (`"""Record one invoke's input; return its ``[start, size]``
group (or None)."""`). Summary plus one paragraph when there's a gotcha, an
ordering constraint, or a rationale — this is the common case. Multi-paragraph is
for modules and top-level API.

**Private functions get a docstring when they aren't self-evident**, and a real
one-liner when they are. Dunders get one when they carry protocol semantics
(`__enter__`, `__exit__`) and none when trivial. Exception classes are
docstring-only — no `__init__`, no `pass`.

---

## Comments

**Explain why, never what.** If the comment paraphrases the code, delete it.
Comment the non-obvious: greenlet and weakref lifetimes, hook ordering, descriptor
rules, autograd aliasing, serialization boundaries. Self-evident code gets
nothing — whole functions here carry no inline comments at all, and that is
correct.

**Name the constraint, then the decision it forces:**

```python
# cat (not in-place) keeps autograd correct for leaves/views and
# avoids aliasing when `edited` is a narrowed view of `full`.
```

**Name the failure mode averted, and the counterfactual where it helps:**

```python
# (nn.Module.__call__ binds `forward` before running its pre-hooks, so
# a controller installed only now — after the input read resumes the worker
# mid-pre-hook — would be too late for the already-bound forward.)
```

**Above-block comments** are capitalized full sentences ending in a period,
wrapped at ~85 columns, typically two to four lines and never more than eight.
They sit above the block, not beside it.

**Trailing comments** are the one exception to that voice: a lowercase fragment
with no period, captioning a single expression.

```python
original = type(module).forward  # unbound; used only for signature metadata
```

**Backtick identifiers** in comments; Sphinx roles (`:meth:`, `:class:`) are
welcome there too. Cross-reference sibling modules by path — `(see
intervention/batching.py)`.

**Comment groups of attributes above the assignment run**, not field by field.

**`__getstate__` and serialization comments always state what travels and what
must not**, and why.

**Section rules use a prose title naming the narrative**, not a category:

```python
# ---------------------------------------------------------------------------
# Compiling the instrumented forward
# ---------------------------------------------------------------------------
```

"Compiling the instrumented forward" tells the reader what they're about to read.
"Private methods" tells them what they can already see.

**`#:` documents module constants and NamedTuple fields.**

---

## Types, imports, and errors

**Annotate everything.** Every parameter, every return — including `-> None` on
`__init__`. `Any` is not a smell in a library that handles arbitrary user values;
use it where it's honest.

**`from __future__ import annotations` goes in every module**, immediately after
the docstring. (`__init__.py` files are the exception.)

**Modern syntax throughout**: `X | None`, `list[str]`, `dict[str, int]`,
`tuple[int, ...]`. Not `Optional`, `List`, `Dict`, `Tuple`, `Union`.

**Imports are relative within the package** (`from ..util import apply`), grouped
stdlib → third-party → local, with `if TYPE_CHECKING:` last. `TYPE_CHECKING` is
for breaking import cycles and for optional third-party types.

**Deferred imports have exactly two justifications, and you say which inline:**

```python
import httpx  # lazy: only needed for actual remote calls
from .iterator import Iterations  # (cycle)
```

**`# noqa` always carries a code, and a reason when the code isn't
self-explanatory:**

```python
except Exception:  # noqa: BLE001 — extension optional; save(value) still works
```

**Raise by default; warn only when the program can proceed with a
degraded-but-sane result** and the user needs to know what it did instead.
Deprecations warn with `DeprecationWarning` and `stacklevel=2`.

**Error messages are terse and state the observed condition.** Quote values with
`{x!r}` rather than hand-rolled quotes. Continue with `;` and a lowercase clause
rather than a second capitalized sentence. Name the user-facing API, not the
internals. No scolding, no "please", no exclamation.

```python
raise ValueError(f"tracer.iter step cannot be negative: {step}")
raise ValueError("Cannot invoke while the model is already running.")
raise KeyError(f"{path!r} was not cached")
```

**Exception variables are `error` or `exception`, never `e`.** Re-wrapping always
chains: `raise RemoteError(f"Failed to send request: {error}") from error`.

**No `assert` in library code** — it vanishes under `-O`, and it isn't error
handling. **No `logging`** — problems surface as exceptions or warnings.

**Guard clauses over nesting.** Return early; keep the happy path unindented.
Reconstruct containers by their own type (`type(data)(...)`) so subclasses
survive.

**`__repr__` is always meaningful**, never the default. Where it helps the person
debugging, it can be a rendered view of the thing itself.

**Instance state is assigned in `__init__`**, annotated inline when the type isn't
obvious from the right-hand side. Class-level attributes are for constants, not
mutable state.

**An unbounded cache must justify never evicting**, in a comment, in terms of what
bounds it in practice.
