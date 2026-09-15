"""Modules standing in for those another pipeline stage owns.

Every rank runs the whole intervention block against the whole tree. On a rank
that does not hold a module, its path carries a :class:`RemoteShell`. Reads and
writes of ``.output``, ``.input`` and ``.inputs`` cross stages through the pull
protocol without touching the module. A call or a parameter has nothing real to
work on here, so the shell raises and names the owning stage. The behavior
lives on the module because a request's envoys are rebuilt on the worker
around the modules the worker holds.
"""

from __future__ import annotations

from typing import Any, NoReturn, Optional

import torch

from vllm.model_executor.models.utils import PPMissingLayer


class RemoteModuleError(RuntimeError):
    """A use of a module that lives on another pipeline stage."""


class RemoteShell(PPMissingLayer):
    """A module another stage holds, as this rank sees it.

    A ``PPMissingLayer`` by type, so vLLM and the graft treat it as the
    placeholder it replaces. ``forward`` raises. An attribute the real module
    has, read off the meta-device copy, raises the same way; any other missing
    attribute raises ``AttributeError`` as on any module.
    """

    def __init__(self, meta: Optional[torch.nn.Module], path: str, owner: Optional[int], rank: int) -> None:
        super().__init__()
        # Kept out of ``_modules`` so the meta copy's parameters are not ours.
        object.__setattr__(self, "_pp_meta", meta)
        self._pp_path = path
        self._pp_owner = owner
        self._pp_rank = rank

    def _remote(self, use: str) -> NoReturn:
        raise RemoteModuleError(
            f"{self._pp_path!r} lives on pipeline stage {self._pp_owner}; {use} is "
            f"not available on stage {self._pp_rank}. Its .output, .input and "
            f".inputs can be read and written from any stage."
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._remote("calling it")

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("_pp_meta")
        if meta is not None and (name in meta._parameters or name in meta._buffers):
            self._remote(f"its attribute .{name}")
        return super().__getattr__(name)


def install_shells(
    model: torch.nn.Module, meta_model: torch.nn.Module, module_map: Any, local_rank: int
) -> list[str]:
    """Replace every module of ``model`` that another stage owns with a
    :class:`RemoteShell`; return the replaced paths.

    Ownership comes from ``module_map``; the meta copy of each replaced module
    comes from ``meta_model``. Children of a replaced module are not visited:
    the graft (:func:`graft_children`) adds them later, as shells too.
    """
    root = module_map.root_path
    meta_modules = dict(meta_model.named_modules())
    modules = dict(model.named_modules())
    installed = []
    for name, module in modules.items():
        if not name or any(name.startswith(f"{done}.") for done in installed):
            continue
        path = f"{root}.{name}"
        if module_map.is_local(path, local_rank):
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        shell = RemoteShell(meta_modules.get(name), path, module_map.get_owning_rank(path), local_rank)
        setattr(parent, attribute, shell)
        installed.append(name)
    return [f"{root}.{name}" for name in installed]


def graft_children(root: Any, meta_model: torch.nn.Module, local_rank: int) -> None:
    """Give every shell envoy under ``root`` the children its meta copy has,
    each wrapped as a shell, so paths under a module another stage owns
    resolve at request deserialization and answer as shells do.
    """
    meta_modules = {f"{root.path}.{name}": module for name, module in meta_model.named_modules()}

    def graft(envoy: Any) -> None:
        module = envoy._module
        if isinstance(module, RemoteShell):
            meta = meta_modules.get(envoy.path)
            if meta is not None:
                for name, child in meta.named_children():
                    envoy._wrap_envoy(
                        name,
                        RemoteShell(child, f"{envoy.path}.{name}", module._pp_owner, local_rank),
                    )
        for child in list(envoy._children):
            graft(child)

    graft(root)
