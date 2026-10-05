"""Module-name aliases: what a ``rename`` key names, and every spelling of a path under them.

``rename={...}`` gives modules extra names. `bind` attaches them to an
[`Envoy`][nnsight.intervention.envoy.Envoy] as it is built, resolving each key
with `named`. `spellings` states the same rule on a path alone, so an
``envoys=`` key written in aliased names can be matched (with `path_ends_with`)
before the aliases bind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .envoy import Envoy


def bind(envoy: Envoy) -> None:
    """Bind each ``rename`` alias as an attribute pointing at the same Envoy.

    For every ``path -> alias(es)`` entry, resolve ``path`` *relative to
    ``envoy``*; if it names a descendant envoy, bind each alias as an attribute on
    ``envoy`` pointing at that same descendant object. Because every envoy in
    the tree runs this, a single-component path like ``"mlp"`` binds wherever
    it resolves (each block that has one), while a multi-component path like
    ``"transformer.h.3.mlp"`` binds only on the envoy it resolves from. A
    leading dot is a no-op — path components are matched by name (an empty
    first component is skipped, mirroring `nnsight.util.fetch_attr`).

    A key may also be a module *class*: ``{Mamba2Mixer: "linear_attn"}``
    binds the alias on every envoy that has exactly one direct child whose
    module is an instance of that class, pointing at that child. This is
    for a tree that holds different modules under one native name (a
    hybrid's ``mixer``: a Mamba-2 mixer on one block, an attention on the
    next), where the standard name follows what the block holds and no
    name-keyed entry can say so.

    A ``*`` component of a dotted key stands for any one entry:
    ``{"*.h": "layers"}`` mounts the ``h`` under whichever child of
    ``envoy`` has one, so the key need not spell the backbone's name
    (``transformer`` here, ``model`` there).

    An alias names one module, so a class or ``*`` key that matches two
    from one envoy is an error, not a guess (see `named`).

    A key that does not resolve here is skipped, not an error: one ``rename``
    is meant to be reusable across architectures that spell the same module
    differently (``{"attn": "att", "self_attn": "att"}``), and on any given
    envoy only one of those spellings exists.

    An alias that would *displace* something is an error, because there is
    no spelling of it that works. Three things can be underneath: another
    alias from an earlier key, a name on the envoy (a child, its own state,
    or an `Envoy` attribute like ``output`` or ``trace``), or a name on the
    wrapped module — an alias lands in ``__dict__`` and so wins over the
    ``__getattr__`` fallthrough, silently shadowing the model's own
    ``config``, ``weight``, ``eval`` and the rest.

    Aliases are ordinary attributes referencing the *same* child object (not
    copies, and not added to `_children`), so ``__getattr__`` needs no
    alias branch, iteration doesn't double-count, and re-pointing the tree on
    dispatch (`_update`, in place) keeps them valid with no rebuild.
    """
    if not envoy._rename:
        return
    for key, aliases in envoy._rename.items():
        matches = named(envoy, key)
        if len(matches) > 1:
            name = key.__name__ if isinstance(key, type) else repr(key)
            raise ValueError(
                f"`rename` key {name} matches {len(matches)} modules under "
                f"`{envoy.path}` ({', '.join(path for path, _ in matches)}); an alias "
                f"names one module, so a key binds only where it matches one. "
                f"Key those by name."
            )
        if not matches:
            continue
        path, target = matches[0]
        for alias in [aliases] if isinstance(aliases, str) else aliases:
            # Already pointing at this very envoy: nothing is displaced.
            # Covers a key aliased to its own name (``{"attn": "attn"}``,
            # which a cross-architecture dict pairs with
            # ``{"self_attn": "attn"}``) and two keys reaching one module
            # through tied weights.
            if envoy.__dict__.get(alias) is target:
                continue

            # What the name would displace, in the order lookup would find
            # it. `hasattr` is right for the module and wrong for the envoy:
            # an eproperty's `__get__` raises outside interleaving, so the
            # envoy's own classes are scanned by namespace instead.
            bound = envoy._aliases.get(alias)
            if bound is not None:
                displaced = f"the alias already bound here from {bound!r}"
            elif alias in envoy.__dict__:
                displaced = "a child module or attribute of that name"
            elif any(alias in vars(klass) for klass in type(envoy).__mro__):
                displaced = f"the `{type(envoy).__name__}.{alias}` attribute"
            elif hasattr(envoy._module, alias):
                # The same test `__getattr__` gates its fallthrough on, so
                # this is the shadowing surface exactly: every name it finds
                # is one `envoy.<name>` answers to today.
                displaced = "an attribute of the wrapped module"
            else:
                displaced = None

            if displaced is not None:
                raise ValueError(
                    f"`rename` alias {alias!r} for {path!r} would shadow "
                    f"{displaced} on `{envoy.path}`. Aliases are bound as plain "
                    f"attributes, so this would make the original unreachable "
                    f"by name while leaving it in the tree. Pick a different "
                    f"alias."
                )

            object.__setattr__(envoy, alias, target)
            envoy._aliases[alias] = path


def named(envoy: Envoy, key: str | type) -> list[tuple[str, Envoy]]:
    """The descendants a ``rename`` key names from ``envoy``, as (relative path, envoy).

    A class key names each direct child whose module is an instance of it.
    A dotted key is walked a component at a time, as `get` walks it, and a
    ``*`` component stands for every entry of the module reached so far:
    ``"*.h"`` names the ``h`` under whichever child has one.
    """
    from .envoy import Envoy

    if isinstance(key, type):
        found = [
            (name, child) for name, child in envoy._child_map.items()
            if isinstance(child._module, key)
        ]
    else:
        found = [("", envoy)]
        for part in key.lstrip(".").split("."):
            step = []
            for path, obj in found:
                if part == "*":
                    if isinstance(obj, Envoy):
                        step += [(f"{path}.{name}", child) for name, child in obj._child_map.items()]
                    continue
                try:
                    step.append((f"{path}.{part}", getattr(obj, part)))
                except AttributeError:
                    continue
            found = step
        found = [(path[1:], obj) for path, obj in found if isinstance(obj, Envoy)]
    # A module the tree holds under two names is one envoy, so one match.
    unique = {id(match): (path, match) for path, match in reversed(found)}
    return list(reversed(unique.values()))


def spellings(
    rename: dict | None, path: str, module: torch.nn.Module | None = None
) -> list[str]:
    """Every spelling of ``path`` under the ``rename`` aliases, the native one first.

    A spelling of a path is a spelling of its parent's path plus the last
    component, or, where a ``rename`` key names the path from an ancestor, a
    spelling of that ancestor's path plus the alias (the rule `bind`
    binds by, so it is known before the aliases bind). Spellings so compose
    through ancestors: under ``{"transformer.h": "layers", "attn": "self_attn"}``,
    ``model.transformer.h.0.attn`` is also ``model.layers.0.self_attn``. A
    class key names ``module`` by type, so it stands for the last component
    only: its alias is a spelling of ``module``, not of what is under it.
    """
    parts = path.split(".")
    if len(parts) == 1:
        return [path]
    parent = ".".join(parts[:-1])
    found = [f"{prefix}.{parts[-1]}" for prefix in spellings(rename, parent)]
    for key, aliases in (rename or {}).items():
        if isinstance(key, type):
            if not isinstance(module, key):
                continue
            ancestor = parent
        else:
            # The key is relative to an ancestor, so it is shorter than the path.
            size = key.lstrip(".").count(".") + 1
            if size >= len(parts) or not path_ends_with(path, key):
                continue
            ancestor = ".".join(parts[:-size])
        heads = spellings(rename, ancestor)
        for alias in [aliases] if isinstance(aliases, str) else aliases:
            found += [f"{prefix}.{alias}" for prefix in heads]
    return found


def path_ends_with(path: str, key: str) -> bool:
    """Whether ``path`` ends with dotted ``key`` component-wise (not substring).

    A ``*`` component of ``key`` matches any one component: ``"layers.*"``
    ends ``model.layers.0`` (every entry of the container), but neither
    ``model.layers`` nor ``model.layers.0.attn``.
    """
    parts = path.split(".")
    key_parts = key.removeprefix(".").split(".")
    return len(key_parts) <= len(parts) and all(
        k in ("*", p) for p, k in zip(parts[-len(key_parts):], key_parts)
    )
