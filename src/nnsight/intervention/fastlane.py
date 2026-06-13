"""Fast-lane safety classifier for isolated traces.

`isolate_mediators()` runs each user intervention in a spawned GPU worker process to
contain footguns (infinite loops, OOM allocs, device-side asserts, host-object pokes,
fs/net/exec). But the worker holds a *weightless* path-only mirror of the model — its
dummy modules have no parameters and no ``forward``. The real interpretability workloads
(logit lens, steering, ablation, activation patching, attribution) all read the host
model's real weights (``F.linear(x, head.weight)``) and call its final-norm / unembed
modules, so **they cannot run in the worker at all**. The fast lane is the tier where the
real weights live: a confirmed-safe intervention runs in-process (the existing daemon-
thread path) at full speed and with full model access, and only unconfirmable code goes
to the worker.

"Confirmed safe" is decided here by a **fail-closed, default-deny** static walk over the
*effective code* — the trace body PLUS every user closure it calls, resolved through the
frame, the function globals, and closure cells (the harness wraps real compute in
``build()`` / ``capture()`` / ``patch()`` closures, so a walk of the ``with`` block alone
would see only an opaque call). The walk emits one of three verdicts:

- ``FAST``    — every node is on the allowlist and every call resolves to a whitelisted
                op, a host module/weight access, an nnsight primitive, or a recursively
                confirmed user function → run in-process.
- ``ISOLATE`` — anything unconfirmable (an unresolved global call, an import, a ``while``
                loop, an unrecoverable closure, an unknown node type) → run in the worker.
- ``REJECT``  — an introspection escape (``__globals__``/``getattr``/``eval``/…) → raise.

The conservative default is ``ISOLATE``: the absence of proof is not proof of safety; only
explicitly whitelisted code reaches ``FAST``.

**Threat model (load-bearing).** This is *not* an adversarial sandbox — a determined
author can defeat any in-process restriction (the pysandbox lesson), so the fast lane is
gated on ``trust="local"`` provenance and disabled for anything deserialized/remote. Under
the relaxed "contain footguns, not adversaries" model it confirms the effective code
(a) introduces no ambient authority (no import, no introspection, no unresolved global
call), (b) has no unbounded loop, (c) writes no host state, and (d) is composed only of
whitelisted ops, host-object access, and recursively confirmed user functions. OOM and
device-side asserts in pure tensor math are knowingly traded to the in-process tier (a
deployment that cannot tolerate them disables the fast lane); a wall-clock watchdog backs
the one loop footgun the static walk cannot bound (a huge ``range``).
"""
from __future__ import annotations

import ast
import ctypes
import inspect
import textwrap
import threading
from dataclasses import dataclass, field

# Verdict tiers.
FAST = "fast"
ISOLATE = "isolate"
REJECT = "reject"

# Modules whose top-level functions are pure compute with no ambient authority. A call
# resolving into one of these is allowed outright.
_SAFE_MODULE_PREFIXES = ("torch", "math", "operator", "numpy")

# torch entry points that DO touch fs/net/JIT despite living under `torch` — never fast.
_BANNED_QUALIFIED = {
    "torch.load", "torch.save", "torch.hub", "torch.jit", "torch.compile",
    "torch.onnx", "torch.multiprocessing", "torch.distributed",
}

# Builtins that are pure / structural — safe to call from confirmed code.
_SAFE_BUILTINS = frozenset({
    "range", "len", "enumerate", "zip", "list", "tuple", "dict", "set", "frozenset",
    "min", "max", "sum", "abs", "round", "sorted", "reversed", "map", "filter",
    "float", "int", "bool", "str", "slice", "isinstance", "issubclass", "all", "any",
    "print", "repr", "iter", "next",
})

# Names / attributes that are an introspection escape — their presence is a REJECT, not an
# isolate: trusted-local author code reaching for these is a footgun (or an attempt to
# break out), and the worker cannot run them either. Every documented in-process escape
# walks one of these (`().__class__.__subclasses__()`, `obj.__globals__`, a fetched
# builtin), which is exactly what pysandbox could not close.
_INTROSPECTION = frozenset({
    "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "vars", "locals", "breakpoint", "memoryview",
})


class FastLaneRejected(Exception):
    """Raised when the classifier finds an introspection escape in a fast-lane-eligible
    trace. The worker cannot run such code either, and under the relaxed footgun model
    trusted-local author code reaching for introspection is a footgun — so fail loudly
    rather than silently route it anywhere."""


class FastLaneTimeout(Exception):
    """Injected by the watchdog into a fast-lane intervention thread that overran the
    wall-clock deadline (a runaway pure-Python loop). It is an ``Exception`` so the
    intervention body's own ``except Exception`` catches it and routes it through the
    normal ``mediator.exception`` path — the host re-raises it to the user, the model
    server is unaffected."""


class Watchdog:
    """Best-effort wall-clock bound on a fast-lane intervention thread.

    The static gate bans ``while`` and unbounded iteration, so the only loop footgun that
    can reach the fast lane is a huge bounded ``range`` or deep recursion. This injects a
    :class:`FastLaneTimeout` into the running thread at its next bytecode if it overruns —
    restoring the loop-containment guarantee that turning isolation on implies. It CANNOT
    preempt a wedged native/CUDA call (only the worker-process kill can); the bounded-loop
    static rule is the primary defense and this is the backstop.
    """

    def __init__(self, deadline_s: float):
        self._deadline = deadline_s
        self._timer = None
        self._ident = None

    def arm(self, thread_ident: int) -> None:
        self._ident = thread_ident
        self._timer = threading.Timer(self._deadline, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        ident = self._ident
        if ident is None:
            return
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(ident), ctypes.py_object(FastLaneTimeout)
        )
        if res > 1:
            # affected more than the target thread — undo to avoid corrupting others
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(ident), None)

    def disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._ident = None


@dataclass
class Verdict:
    """The classifier's decision for one mediator's intervention."""

    tier: str                       # FAST | ISOLATE | REJECT
    reason: str
    differentiate: bool = False     # the effective code opens a `with x.backward():`
    touches_host_weights: bool = False
    in_place: bool = False          # an in-place write on a delivered boundary value
    _seen: set = field(default_factory=set, repr=False)  # visited code objects (recursion guard)

    @property
    def fast(self) -> bool:
        return self.tier == FAST


# Sentinel raised internally to short-circuit the walk on the first disqualifying node.
class _Stop(Exception):
    def __init__(self, tier: str, reason: str):
        self.tier = tier
        self.reason = reason


def classify(mediator) -> Verdict:
    """Classify a mediator's intervention for the fast lane. Always returns a Verdict
    (never raises); a REJECT tier is surfaced as a raised error by the caller."""
    v = Verdict(tier=FAST, reason="confirmed: effective code is all whitelisted ops")
    try:
        nodes = _root_nodes(mediator)
        if nodes is None:
            return Verdict(ISOLATE, "could not recover the trace body source/AST")
        ns = _root_namespace(mediator)
        walker = _Walker(v)
        walker.walk(nodes, ns, set())
    except _Stop as s:
        return Verdict(s.tier, s.reason, v.differentiate, v.touches_host_weights, v.in_place)
    except Exception as e:  # noqa: BLE001 — any analysis failure is fail-closed
        return Verdict(ISOLATE, f"classifier could not confirm safety: {type(e).__name__}: {e}")
    return v


def classify_callable(fn) -> Verdict:
    """Classify an arbitrary callable's body with the same engine `classify` uses on a
    trace body. The effective-code walk resolves names through ``fn``'s globals and
    closure cells. Used by the fast-lane tests to confirm verdicts on varied/renamed
    module structures without standing up a model."""
    v = Verdict(tier=FAST, reason="confirmed: effective code is all whitelisted ops")
    try:
        _Walker(v)._recurse_fn(fn, depth=0)
    except _Stop as s:
        return Verdict(s.tier, s.reason, v.differentiate, v.touches_host_weights, v.in_place)
    except Exception as e:  # noqa: BLE001
        return Verdict(ISOLATE, f"classifier could not confirm safety: {type(e).__name__}: {e}")
    return v


def _root_nodes(mediator):
    """The statements of the trace body, preferring the live AST node, re-parsing the
    captured source otherwise."""
    info = mediator.info
    node = getattr(info, "node", None)
    if isinstance(node, ast.With):
        return list(node.body)
    src = info.source
    if not src:
        return None
    text = textwrap.dedent("".join(src))
    tree = ast.parse(text)
    body = tree.body
    if len(body) == 1 and isinstance(body[0], ast.With):
        return list(body[0].body)
    return body


def _root_namespace(mediator) -> dict:
    """Name → object map for the trace body: the capturing frame's locals/globals plus
    the compiled intervention's module globals."""
    ns = {}
    fn = mediator.intervention
    g = getattr(fn, "__globals__", None)
    if isinstance(g, dict):
        ns.update(g)
    frame = getattr(mediator.info, "frame", None)
    fl = getattr(frame, "f_globals", None)
    if isinstance(fl, dict):
        ns.update(fl)
    fl = getattr(frame, "f_locals", None)
    if isinstance(fl, dict):
        ns.update(fl)
    return ns


def _fn_namespace(fn) -> dict:
    """Name → object map for a resolved user function: its globals plus closure cells."""
    ns = {}
    g = getattr(fn, "__globals__", None)
    if isinstance(g, dict):
        ns.update(g)
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is not None and closure:
        for name, cell in zip(code.co_freevars, closure):
            try:
                ns[name] = cell.cell_contents
            except ValueError:
                pass  # an empty cell (recursive def not yet bound)
    return ns


class _Walker:
    """Default-deny AST walk. Raises ``_Stop`` on the first ISOLATE/REJECT node; falling
    off the end means every node was confirmed FAST."""

    MAX_DEPTH = 12

    def __init__(self, verdict: Verdict):
        self.v = verdict

    # --- entry points ---------------------------------------------------------
    def walk(self, nodes, ns: dict, local_names: set, depth: int = 0):
        for n in nodes:
            self._stmt(n, ns, local_names, depth)

    def _recurse_fn(self, fn, depth: int):
        """Walk a resolved user function's body. Unrecoverable source → ISOLATE."""
        code = getattr(fn, "__code__", None)
        if code is None or code in self.v._seen:
            return
        if depth >= self.MAX_DEPTH:
            raise _Stop(ISOLATE, "intervention call graph is deeper than the fast-lane bound")
        self.v._seen.add(code)
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            raise _Stop(ISOLATE, f"could not recover source for `{getattr(fn, '__name__', fn)}`")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            raise _Stop(ISOLATE, f"could not parse source for `{getattr(fn, '__name__', fn)}`")
        target = tree.body[0]
        ns = _fn_namespace(fn)
        # The function's own parameters are locals we cannot resolve to objects (they are
        # bound from the call site) — record them so a call/use of a param is treated as a
        # host-object access under trust=local, not as an unknown global.
        params = set()
        node = target
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = _arg_names(node.args)
            self.walk(node.body, ns, params, depth + 1)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Lambda):
            lam = node.value
            params = _arg_names(lam.args)
            self._expr(lam.body, ns, params, depth + 1)
        else:
            # getsource gave us the surrounding statement (common for lambdas passed as
            # args); find the first Lambda anywhere in it.
            lam = next((d for d in ast.walk(tree) if isinstance(d, ast.Lambda)), None)
            if lam is None:
                raise _Stop(ISOLATE, f"could not isolate the body of `{getattr(fn, '__name__', fn)}`")
            params = _arg_names(lam.args)
            self._expr(lam.body, ns, params, depth + 1)

    # --- statements -----------------------------------------------------------
    def _stmt(self, n, ns, loc, depth):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            self._check_targets(n, ns, loc, depth)
            if n.value is not None:
                self._expr(n.value, ns, loc, depth)
            # bind assigned simple names as locals (so later uses are not "unknown global")
            for t in _assigned_names(n):
                loc.add(t)
        elif isinstance(n, ast.Expr):
            self._expr(n.value, ns, loc, depth)
        elif isinstance(n, ast.Return):
            if n.value is not None:
                self._expr(n.value, ns, loc, depth)
        elif isinstance(n, ast.If):
            self._expr(n.test, ns, loc, depth)
            self.walk(n.body, ns, loc, depth)
            self.walk(n.orelse, ns, loc, depth)
        elif isinstance(n, ast.For):
            # bounded by its iterable; an unbounded generator would be a user fn we walk,
            # and a huge `range` is backed by the wall-clock watchdog.
            for t in _target_names(n.target):
                loc.add(t)
            self._expr(n.iter, ns, loc, depth)
            self.walk(n.body, ns, loc, depth)
            self.walk(n.orelse, ns, loc, depth)
        elif isinstance(n, ast.With):
            for item in n.items:
                self._with_item(item, ns, loc, depth)
            self.walk(n.body, ns, loc, depth)
        elif isinstance(n, ast.Raise):
            # `raise ValueError(...)` is a normal guard; walk its expressions.
            for child in (n.exc, n.cause):
                if child is not None:
                    self._expr(child, ns, loc, depth)
        elif isinstance(n, ast.Pass):
            pass
        elif isinstance(n, (ast.FunctionDef, ast.Lambda)):
            # a def/lambda inside the body is a value; bind its name, walk when called
            if isinstance(n, ast.FunctionDef):
                loc.add(n.name)
                loc |= _arg_names(n.args)
                self.walk(n.body, ns, loc, depth)
        elif isinstance(n, ast.While):
            raise _Stop(ISOLATE, "a `while` loop cannot be statically bounded — isolated")
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            raise _Stop(ISOLATE, "an `import` introduces ambient authority — isolated")
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            raise _Stop(REJECT, "`global`/`nonlocal` rebinds shared state — rejected")
        elif isinstance(n, ast.Try):
            raise _Stop(ISOLATE, "a `try` block is not confirmable — isolated")
        else:
            raise _Stop(ISOLATE, f"unsupported statement `{type(n).__name__}` — isolated")

    def _with_item(self, item, ns, loc, depth):
        ctx = item.context_expr
        # detect `with <x>.backward():` (closure-aware — this is the fix for the old
        # `.backward(` substring that missed backwards hidden in a closure).
        if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute) \
                and ctx.func.attr == "backward":
            self.v.differentiate = True
        self._expr(ctx, ns, loc, depth)
        if item.optional_vars is not None:
            for t in _target_names(item.optional_vars):
                loc.add(t)

    def _check_targets(self, n, ns, loc, depth):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
        for t in targets:
            self._store_target(t, ns, loc, depth)

    def _store_target(self, t, ns, loc, depth):
        if isinstance(t, ast.Name):
            return  # binding a local name — fine
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._store_target(e, ns, loc, depth)
            return
        if isinstance(t, ast.Subscript):
            # `hidden[:] = ...` — in-place write into a delivered/derived tensor. Correct
            # in-process (shared memory); it is the steering default and is *safer* on the
            # fast lane than under isolation (where clone-on-receive makes it a silent no-op).
            self.v.in_place = True
            self._expr(t.value, ns, loc, depth)
            return
        if isinstance(t, ast.Attribute):
            # `x.output = ...` / `x.input = ...` is the nnsight boundary write (a SWAP),
            # allowed. Any other attribute store mutates host state visible to sibling
            # mediators — isolate.
            if t.attr in ("output", "input", "inputs", "grad"):
                self._expr(t.value, ns, loc, depth)
                return
            if t.attr.startswith("_"):
                raise _Stop(REJECT, f"writing dunder/private attribute `{t.attr}` — rejected")
            raise _Stop(ISOLATE, f"writing host attribute `.{t.attr}` mutates shared state — isolated")
        # computed target (e.g. starred) — be conservative
        raise _Stop(ISOLATE, f"unsupported assignment target `{type(t).__name__}` — isolated")

    # --- expressions ----------------------------------------------------------
    def _expr(self, e, ns, loc, depth):
        if isinstance(e, ast.Call):
            self._call(e, ns, loc, depth)
        elif isinstance(e, ast.Attribute):
            self._attribute(e, ns, loc, depth)
        elif isinstance(e, ast.Name):
            if e.id in _INTROSPECTION:
                raise _Stop(REJECT, f"name `{e.id}` is an introspection escape — rejected")
        elif isinstance(e, ast.Subscript):
            self._subscript(e, ns, loc, depth)
        elif isinstance(e, ast.Constant):
            pass
        elif isinstance(e, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            for child in ast.iter_child_nodes(e):
                if isinstance(child, ast.expr):
                    self._expr(child, ns, loc, depth)
        elif isinstance(e, (ast.List, ast.Tuple, ast.Set)):
            for elt in e.elts:
                self._expr(elt, ns, loc, depth)
        elif isinstance(e, ast.Dict):
            for k in e.keys:
                if k is not None:
                    self._expr(k, ns, loc, depth)
            for val in e.values:
                self._expr(val, ns, loc, depth)
        elif isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            self._comprehension(e, ns, loc, depth)
        elif isinstance(e, ast.Slice):
            for child in (e.lower, e.upper, e.step):
                if child is not None:
                    self._expr(child, ns, loc, depth)
        elif isinstance(e, ast.IfExp):
            self._expr(e.test, ns, loc, depth)
            self._expr(e.body, ns, loc, depth)
            self._expr(e.orelse, ns, loc, depth)
        elif isinstance(e, ast.Starred):
            self._expr(e.value, ns, loc, depth)
        elif isinstance(e, (ast.JoinedStr,)):
            for val in e.values:
                if isinstance(val, ast.FormattedValue):
                    self._expr(val.value, ns, loc, depth)
        elif isinstance(e, ast.Lambda):
            inner = loc | _arg_names(e.args)
            self._expr(e.body, ns, inner, depth)
        elif isinstance(e, ast.Constant):
            pass
        else:
            raise _Stop(ISOLATE, f"unsupported expression `{type(e).__name__}` — isolated")

    def _comprehension(self, e, ns, loc, depth):
        inner = set(loc)
        for gen in e.generators:
            for t in _target_names(gen.target):
                inner.add(t)
            self._expr(gen.iter, ns, inner, depth)
            for cond in gen.ifs:
                self._expr(cond, ns, inner, depth)
        if isinstance(e, ast.DictComp):
            self._expr(e.key, ns, inner, depth)
            self._expr(e.value, ns, inner, depth)
        else:
            self._expr(e.elt, ns, inner, depth)

    def _subscript(self, e, ns, loc, depth):
        # a dunder/private string key is an introspection escape (`obj["__globals__"]`).
        sl = e.slice
        key = sl.value if isinstance(sl, ast.Constant) else None
        if isinstance(key, str) and key.startswith("_"):
            raise _Stop(REJECT, f"subscript key `{key}` is an introspection escape — rejected")
        self._expr(e.value, ns, loc, depth)
        if isinstance(sl, ast.expr):
            self._expr(sl, ns, loc, depth)

    def _attribute(self, e, ns, loc, depth):
        # reading an attribute is fine (incl. `.weight`, `.output`, `.shape`); reading a
        # dunder/introspection attribute is an escape.
        if e.attr in _INTROSPECTION or (e.attr.startswith("__") and e.attr.endswith("__")):
            raise _Stop(REJECT, f"attribute `.{e.attr}` is an introspection escape — rejected")
        if e.attr == "weight" or e.attr == "bias":
            self.v.touches_host_weights = True
        self._expr(e.value, ns, loc, depth)

    def _call(self, e, ns, loc, depth):
        # walk args first (they may themselves disqualify)
        for a in e.args:
            self._expr(a, ns, loc, depth)
        for kw in e.keywords:
            self._expr(kw.value, ns, loc, depth)

        func = e.func
        if isinstance(func, ast.Attribute):
            self._call_attribute(func, ns, loc, depth)
        elif isinstance(func, ast.Name):
            self._call_name(func.id, ns, loc, depth)
        else:
            # a computed callable: `(a or b)(x)`, `factory()(x)` — unconfirmable target
            self._expr(func, ns, loc, depth)
            raise _Stop(ISOLATE, "call to a computed/dynamic target — isolated")

    def _call_attribute(self, func: ast.Attribute, ns, loc, depth):
        attr = func.attr
        if attr in _INTROSPECTION or (attr.startswith("__") and attr.endswith("__")):
            raise _Stop(REJECT, f"method `.{attr}()` is an introspection escape — rejected")
        # is this a banned qualified op like torch.load / torch.save / torch.hub.*?
        qual = _attr_chain(func)
        if qual is not None:
            for banned in _BANNED_QUALIFIED:
                if qual == banned or qual.startswith(banned + "."):
                    raise _Stop(ISOLATE, f"`{qual}` touches fs/net/JIT — isolated")
            # if it resolves into a safe module, fine; if into a banned module, isolate
            head = qual.split(".", 1)[0]
            obj = _lookup(head, ns, loc)
            if inspect.ismodule(obj) and _module_is_safe(obj):
                self._expr(func.value, ns, loc, depth)
                return
        # otherwise a method call on a receiver (tensor/envoy/host object). Method-name
        # is not an escape (checked above); the receiver is walked. Allowed under
        # trust=local — `.clone()`, `.save()`, `.sum()`, `.backward()`, `.to(...)` etc.
        self._expr(func.value, ns, loc, depth)

    def _call_name(self, name: str, ns, loc, depth):
        if name in _INTROSPECTION:
            raise _Stop(REJECT, f"`{name}()` is an introspection escape — rejected")
        if name in _SAFE_BUILTINS:
            return
        if name in loc:
            return  # a local/param: a host object or fn passed in at a walked call site
        obj = _lookup(name, ns, None)
        if obj is _MISSING:
            raise _Stop(ISOLATE, f"call to unresolved name `{name}` — unknown authority, isolated")
        kind = _classify_obj(obj)
        if kind == "op" or kind == "nnsight" or kind == "host" or kind == "builtin_ok":
            if kind == "host":
                self.v.touches_host_weights = self.v.touches_host_weights  # host call is fine
            return
        if kind == "banned":
            raise _Stop(REJECT, f"`{name}()` is an introspection escape — rejected")
        if kind == "userfn":
            self._recurse_fn(obj, depth)
            return
        # unknown object type bound to a resolvable name — be conservative
        raise _Stop(ISOLATE, f"call to `{name}` ({type(obj).__name__}) is not confirmable — isolated")


# --------------------------------------------------------------------------- #
# object / name resolution helpers                                            #
# --------------------------------------------------------------------------- #
_MISSING = object()


def _lookup(name, ns: dict, loc):
    if loc is not None and name in loc:
        return _MISSING  # a local: caller treats specially
    if name in ns:
        return ns[name]
    import builtins
    if hasattr(builtins, name):
        return getattr(builtins, name)
    return _MISSING


def _module_is_safe(mod) -> bool:
    modname = getattr(mod, "__name__", "")
    return any(modname == p or modname.startswith(p + ".") for p in _SAFE_MODULE_PREFIXES)


def _classify_obj(obj) -> str:
    import builtins
    if obj is _MISSING:
        return "unknown"
    # introspection builtins resolved as objects
    if getattr(obj, "__name__", None) in _INTROSPECTION and inspect.isbuiltin(obj):
        return "banned"
    mod = getattr(obj, "__module__", None) or ""
    if mod == "builtins":
        nm = getattr(obj, "__name__", None)
        if nm in _INTROSPECTION:
            return "banned"
        if nm in _SAFE_BUILTINS:
            return "builtin_ok"
        return "unknown"  # an unlisted builtin (open/input/…) → isolate, not reject
    # pure-compute libraries
    if any(mod == p or mod.startswith(p + ".") for p in _SAFE_MODULE_PREFIXES):
        return "op"
    # nnsight primitives (Envoy, eproperty, tracer methods, save, ...)
    if mod == "nnsight" or mod.startswith("nnsight."):
        return "nnsight"
    # a host object: Envoy / nn.Module / Tensor / Backend — calling it runs the real model
    if _is_host_object(obj):
        return "host"
    # any other user-defined callable → recurse into its source
    if inspect.isfunction(obj) or inspect.ismethod(obj) or isinstance(obj, type(_lookup)):
        return "userfn"
    if inspect.isfunction(getattr(obj, "__call__", None)):
        return "userfn"
    return "unknown"


def _is_host_object(obj) -> bool:
    try:
        import torch.nn as nn
        import torch
        if isinstance(obj, (nn.Module, torch.Tensor)):
            return True
    except Exception:  # noqa: BLE001
        pass
    cls = type(obj)
    chain = {c.__name__ for c in cls.__mro__}
    return bool(chain & {"Envoy", "Backend"})


def _attr_chain(node):
    """`torch.nn.functional.linear` → that dotted string; None if not a pure attr chain."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _arg_names(args: ast.arguments) -> set:
    names = set()
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in group:
            names.add(a.arg)
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _target_names(target) -> set:
    names = set()
    for n in ast.walk(target):
        if isinstance(n, ast.Name):
            names.add(n.id)
    return names


def _assigned_names(n) -> set:
    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
    names = set()
    for t in targets:
        for x in ast.walk(t):
            if isinstance(x, ast.Name):
                names.add(x.id)
    return names
