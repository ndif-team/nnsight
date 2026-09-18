"""Reads whose value the block only saves.

Every stage runs the same block, so a value that a block reads and does nothing
with but save is saved on the stage that holds it too. Such a read needs no
transfer while the run is on: the reading stage binds a :class:`Deferred` in
its place and the owning stage skips the push, and when the request's values
are collected from every stage the placeholder is filled from the owner's copy
of the same name (`fill_saves`).

Which reads those are is decided from the block's source before it runs, the
same way on every stage (`deferrable_lines`). The rule is conservative: a
statement that saves a read directly (``h = m.layer.output.save()``, with ``h``
never used again), or appends one to a container the block saved and only
appends to (``kept = nnsight.save([]); kept.append(m.layer.output[0])``).
Anything else crosses the wire as it did.
"""

from __future__ import annotations

import ast
import functools
from typing import Any, Optional

from torch.utils._pytree import tree_flatten, tree_unflatten

# The envoy properties a read is made through.
READ_ATTRS = frozenset({"output", "input", "inputs", "logits", "samples"})


class Deferred:
    """Stands in for a value another stage holds and the block only saves.

    Attributes:
        provider: The location read.
        step: The step of the run it was read at.
    """

    __slots__ = ("provider", "step")

    def __init__(self, provider: str, step: int) -> None:
        self.provider = provider
        self.step = step

    def __getitem__(self, index: Any) -> "Deferred":
        # The owner saves the indexed value at the same place; the index is
        # already applied there.
        return self

    def __repr__(self) -> str:
        return f"Deferred({self.provider!r}, step={self.step})"


def _is_read(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr in READ_ATTRS


def _read_of(node: ast.AST) -> Optional[ast.Attribute]:
    """The read at the root of ``node``, which is a read optionally subscripted
    with constants; ``None`` when ``node`` has any other shape, or when the
    read's own base contains another read."""
    while isinstance(node, ast.Subscript):
        if any(not isinstance(n, (ast.Constant, ast.Slice, ast.Tuple, ast.UnaryOp, ast.USub)) for n in ast.walk(node.slice)):
            return None
        node = node.value
    if not _is_read(node):
        return None
    if any(_is_read(n) for n in ast.walk(node.value)):
        return None
    return node


def _save_call(node: ast.AST) -> Optional[ast.AST]:
    """``x`` for a node of the form ``x.save()``, ``save(x)`` or ``nnsight.save(x)``."""
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "save":
        if not node.args:
            return func.value
        if len(node.args) == 1 and isinstance(func.value, ast.Name):
            return node.args[0]
    if isinstance(func, ast.Name) and func.id == "save" and len(node.args) == 1:
        return node.args[0]
    return None


def _loads(tree: ast.AST) -> dict[str, list[ast.Name]]:
    loads: dict[str, list[ast.Name]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.setdefault(node.id, []).append(node)
    return loads


def _append_only(name: str, tree: ast.AST, loads: dict) -> bool:
    """Whether ``name`` is bound once, by a ``.save()`` call, and otherwise only
    ever appears as the receiver of ``name.append(...)`` statements."""
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
        and any(isinstance(t, ast.Name) and t.id == name for t in (node.targets if isinstance(node, ast.Assign) else [node.target]))
    ]
    if len(stores) != 1 or not isinstance(stores[0], ast.Assign) or _save_call(stores[0].value) is None:
        return False
    receivers = {
        id(node.func.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == name
    }
    return all(id(use) in receivers for use in loads.get(name, ()))


@functools.lru_cache(maxsize=256)
def deferrable_lines(source: str) -> frozenset[int]:
    """The lines of ``source`` whose statement reads a location and only saves it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    loads = _loads(tree)
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        read = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # h = <read>.save(), h never read again
            saved = _save_call(node.value)
            if saved is not None and node.targets[0].id not in loads:
                read = _read_of(saved)
        elif isinstance(node, ast.Expr):
            saved = _save_call(node.value)
            if saved is not None:
                # <read>.save() bare
                read = _read_of(saved)
            elif (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "append"
                and isinstance(node.value.func.value, ast.Name)
                and len(node.value.args) == 1
                and not node.value.keywords
                and _append_only(node.value.func.value.id, tree, loads)
            ):
                # kept.append(<read>), kept saved and only appended to
                read = _read_of(node.value.args[0])
        if read is not None:
            lines.update(range(node.lineno, node.end_lineno + 1))
    return frozenset(lines)


def fill_saves(reports: list[dict]) -> dict:
    """One ``name -> value`` from every stage's saved names, in stage order.

    The first stage that saved a name gives its value; a :class:`Deferred` in
    it (at any depth of a container) is filled from the first later stage whose
    value for that name holds a real value at the same place.
    """
    merged: dict[str, Any] = {}
    for report in reports:
        for name, value in report.items():
            merged.setdefault(name, value)
    for name, value in merged.items():
        leaves, spec = tree_flatten(value)
        if not any(isinstance(leaf, Deferred) for leaf in leaves):
            continue
        others = [tree_flatten(report[name])[0] for report in reports if name in report]
        for index, leaf in enumerate(leaves):
            if not isinstance(leaf, Deferred):
                continue
            for other in others:
                if index < len(other) and not isinstance(other[index], Deferred):
                    leaves[index] = other[index]
                    break
            else:
                raise RuntimeError(
                    f"{name!r} holds {leaf}, which no stage saved a value for: "
                    "the block did not reach that save on the stage holding the location"
                )
        merged[name] = tree_unflatten(leaves, spec)
    return merged
