"""Modules standing in for those another pipeline stage owns.

Every rank runs the whole intervention block against the whole tree. On a rank
that does not hold a module, its path carries a :class:`RemoteShell`. Reads and
writes of ``.output``, ``.input`` and ``.inputs`` cross stages through the
interleaver without touching the module. ``param(name)`` asks the owner for the
parameter over the link and returns it as a real tensor here, or takes it from
the module's state when that is already kept. A call asks the owner for the
module's state and runs the meta copy's forward on it through
``torch.func.functional_call``, so the block gets the same result the owner
computes and the copy is not touched. A fetched parameter or state is kept
while the requests that used it run and dropped when they finish; the head's
state is fetched once at load and kept for the engine's life, since it is the
one large weight blocks read across stages (1.45 GiB and 3 s per fetch at
14B). The parameter as a plain attribute has nothing real to work on here, so
the shell raises and names the owning stage. The behavior lives on the module
because a request's envoys are rebuilt on the worker around the modules the
worker holds.
"""

from __future__ import annotations

from typing import Any, Callable, NoReturn, Optional

import torch
from torch.utils._pytree import tree_map
from vllm.model_executor.models.utils import PPMissingLayer

PARAM_MARK = ".param."
STATE_MARK = ".state"


class RemoteModuleError(RuntimeError):
    """A use of a module that lives on another pipeline stage."""


class RemoteShell(PPMissingLayer):
    """A module another stage holds, as this rank sees it.

    A ``PPMissingLayer`` by type, so vLLM and the graft treat it as the
    placeholder it replaces. ``forward`` runs the meta copy on the owner's
    state. An attribute the real module has, read off the meta-device copy,
    raises naming the owner; any other missing attribute raises
    ``AttributeError`` as on any module.
    """

    def __init__(
        self,
        meta: Optional[torch.nn.Module],
        path: str,
        owner: Optional[int],
        rank: int,
        link: Any = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        # Kept out of ``_modules`` so the meta copy's parameters are not ours.
        object.__setattr__(self, "_pp_meta", meta)
        self._pp_path = path
        self._pp_owner = owner
        self._pp_rank = rank
        self._pp_link = link
        self._pp_device = device

    def _remote(self, use: str, instead: str = "") -> NoReturn:
        raise RemoteModuleError(
            f"{self._pp_path!r} lives on pipeline stage {self._pp_owner}; {use} is "
            f"not available on stage {self._pp_rank}. Its .output, .input and "
            f".inputs can be read and written from any stage."
            + (f" {instead}." if instead else "")
        )

    def _fetch(self, provider: str) -> Any:
        """``provider`` from the owner, kept on this rank (see `Kept`): for the
        request whose block asked, or for the engine's life when asked outside
        any request, as the head is at load."""
        from ...intervention.interleaver import Mediator

        try:
            req = Mediator.current(provider).pp_req
        except ValueError:
            # No block is running: asked at load, or from a plain call.
            req = None
        link = self._pp_link
        if provider in link.kept:
            return link.kept.get(provider, req)
        value = link.request(self._pp_owner, provider)
        if self._pp_device is not None:
            value = tree_map(lambda t: t.to(self._pp_device) if isinstance(t, torch.Tensor) else t, value)
        link.kept.put(provider, value, req)
        return value

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the meta copy's forward here on the owner's state.

        ``functional_call`` runs the forward with the state's tensors standing
        in for the copy's parameters and buffers for this call only; the
        copy's own parameter objects, and whatever vLLM stamped on them, are
        left as they are. Under tensor parallelism the state is the column
        peer's shard and the meta copy is built at the same shard shapes, so
        the module's own collectives run in this stage's group as they do on
        the owner. A buffer the module keeps out of its state dict is not
        carried over, so a module that computes from one has to be called on
        its owner.
        """
        meta = self.__dict__.get("_pp_meta")
        if meta is None or self._pp_link is None:
            self._remote("calling it")
        state = self._fetch(f"{self._pp_path}{STATE_MARK}")
        return torch.func.functional_call(meta, state, args, kwargs)

    def _nnsight_parameter(self, name: str) -> torch.Tensor:
        """``param(name)``: the parameter fetched from the owner's module."""
        meta = self.__dict__.get("_pp_meta")
        if meta is None or (name not in meta._parameters and name not in meta._buffers):
            raise AttributeError(f"{self._pp_path!r} has no parameter or buffer named {name!r}")
        if self._pp_link is None:
            self._remote(f"its parameter {name!r}")
        # A state already kept here (the head's, fetched at load, or a call's
        # earlier in this request) holds the parameter; asking the owner again
        # would move the same bytes twice. A buffer kept out of the state dict
        # is not in it and is asked for by name.
        state = f"{self._pp_path}{STATE_MARK}"
        if state in self._pp_link.kept and name in (held := self._fetch(state)):
            pulled = held[name]
        else:
            pulled = self._fetch(f"{self._pp_path}{PARAM_MARK}{name}")
        # The fetch carries this rank's column peer's shard; the meta copy's
        # parameter carries the sharding stamps that say how to gather it.
        from .envoys import _whole_parameter

        reference = getattr(meta, name)
        for stamp in ("output_dim", "input_dim"):
            if hasattr(reference, stamp):
                setattr(pulled, stamp, getattr(reference, stamp))
        return _whole_parameter(meta, pulled)

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("_pp_meta")
        if meta is not None and (name in meta._parameters or name in meta._buffers):
            self._remote(f"its attribute .{name}", f"param({name!r}) pulls it from the owner")
        return super().__getattr__(name)


def install_shells(
    model: torch.nn.Module,
    meta_model: torch.nn.Module,
    module_map: Any,
    local_rank: int,
    link: Any = None,
    device: Optional[torch.device] = None,
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
        shell = RemoteShell(
            meta_modules.get(name), path, module_map.get_owning_rank(path), local_rank, link, device
        )
        setattr(parent, attribute, shell)
        installed.append(name)
    return [f"{root}.{name}" for name in installed]


def graft_children(
    root: Any,
    meta_model: torch.nn.Module,
    local_rank: int,
    link: Any = None,
    device: Optional[torch.device] = None,
) -> None:
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
                        RemoteShell(child, f"{envoy.path}.{name}", module._pp_owner, local_rank, link, device),
                    )
        for child in list(envoy._children):
            graft(child)

    graft(root)


def resolver(model: torch.nn.Module, root: str) -> Callable[[str], Any]:
    """The owner's side of ``param(name)`` and of a call: ``provider -> value``
    read from ``model``'s own modules, for the link to answer with.

    ``{path}.param.{name}`` is that parameter or buffer; ``{path}.state`` is
    the module's state dict.
    """
    modules = {f"{root}.{name}": module for name, module in model.named_modules() if name}

    def held(path: str) -> torch.nn.Module:
        module = modules.get(path)
        if module is None or isinstance(module, RemoteShell):
            raise AttributeError(f"{path!r} is not held on this rank")
        return module

    def resolve(provider: str) -> Any:
        if provider.endswith(STATE_MARK):
            return dict(held(provider[: -len(STATE_MARK)]).state_dict())
        path, mark, name = provider.rpartition(PARAM_MARK)
        if not mark:
            raise AttributeError(f"{provider!r} names neither a parameter nor a module's state")
        value = getattr(held(path), name, None)
        if not isinstance(value, torch.Tensor):
            raise AttributeError(f"{path!r} has no parameter or buffer named {name!r}")
        return value

    return resolve
