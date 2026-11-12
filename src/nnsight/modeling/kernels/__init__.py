"""The kernels transformers models call, and how each one is found.

`CATALOG` is the list of supported kernels: one
[`Spec`][nnsight.intervention.kernel.Spec] row per kernel function, naming its
site family, its serve function, and a locator that finds the function and the
place the model looks it up. `register` runs the catalog against one model when
a `TransformersModel` is built. The registry is process-wide, so a kernel
registered for one model serves every model that calls it.

transformers reaches its kernels in three ways, one locator each:

* `eager_functions`: under ``attn_implementation="eager"``, an attention module
  passes its modeling module's own ``*eager_attention_forward`` global to the
  attention interface on each call.
* `experts_forward`: the experts wrapper looks its implementation up in
  ``ALL_EXPERTS_FUNCTIONS`` on each call.
* `hub_kernel`: recurrent kernels are called through
  ``use_kernel_func_from_hub_with_fallback`` wrappers, which call whatever their
  ``implementation`` closure cell holds: transformers' Python reference, or an
  installed library's kernel.

Readers are in the sibling modules: eager.py, recurrent.py and experts.py.
PyTorch's SDPA has its own one-row catalog in intervention/attention.py.
"""

from __future__ import annotations

import sys
from types import CellType, FunctionType, ModuleType
from typing import Callable, Iterable, NamedTuple

import torch

from ...intervention.kernel import Attr, Item, Spec, register_catalog, unshimmed
from . import eager, experts, recurrent

Located = Iterable[tuple[Callable, tuple[Attr, ...]]]


class Modeling(NamedTuple):
    """What the locators search for one model: its modeling modules and their hub-kernel wrappers."""

    #: The Python modules that define the classes of the model's submodules.
    modules: list[ModuleType]
    #: One dict per hub-kernel wrapper in those modules, mapping each closure variable name to its cell.
    wrappers: list[dict[str, CellType]]


def modeling(root: torch.nn.Module) -> Modeling:
    """Collect the modeling modules of ``root``'s submodules and the hub-kernel wrappers they define."""
    # A dynamically exec'd class has no entry in `sys.modules`.
    modules = list({sys.modules.get(type(sub).__module__) for sub in root.modules()} - {None})
    return Modeling(modules, [cells for module in modules for cells in _hub_wrappers(module)])


def _hub_wrappers(module: ModuleType) -> list[dict[str, CellType]]:
    """Inspect all ``use_kernel_func_from_hub_with_fallback`` wrapper closures defined in ``module``.

    Each wrapper is identified by its code object's name (``"wrapped"``) and file
    (``transformers.integrations.hub_kernels``). Each entry in the returned list
    maps a closure variable name (``func_name``, ``package``, ``torch_function``,
    ``implementation``, etc.) to its closure cell, allowing callers to read
    ``cell_contents`` or pass the cell as a holder to `register`.
    """
    from transformers.integrations import hub_kernels

    found = []
    for value in vars(module).values():
        if not (
            isinstance(value, FunctionType)
            and value.__code__.co_name == "wrapped"
            and value.__code__.co_filename == hub_kernels.__file__
        ):
            continue
        found.append(dict(zip(value.__code__.co_freevars, value.__closure__)))
    return found


def eager_functions(context: Modeling) -> Located:
    """Locate every eager attention function defined in the model's modeling modules.

    `register` adds the function's own module global as its holder, which is the
    binding the attention module passes on each call.
    """
    for module in context.modules:
        for value in list(vars(module).values()):
            # A trace open while this model is built has already put shims in the globals.
            fn = unshimmed(value)
            if eager.is_eager_forward(fn):
                yield fn, ()


def experts_forward(name: str, key: str) -> Callable[[Modeling], Located]:
    """Locate the experts forward ``name``, held at ``key`` of transformers' experts-implementation mapping.

    ``ALL_EXPERTS_FUNCTIONS.__setitem__`` writes a local override entry, so the
    holder is the underlying entry in ``_global_mapping``. transformers releases
    before 5 have no ``transformers.integrations.moe`` and locate nothing.
    """

    def locate(context: Modeling) -> Located:
        try:
            from transformers.integrations import moe
        except ImportError:
            return
        yield unshimmed(vars(moe)[name]), (Item(moe.ALL_EXPERTS_FUNCTIONS._global_mapping, key),)

    return locate


def hub_kernel(name: str, package: str | None = None) -> Callable[[Modeling], Located]:
    """Locate what the hub-kernel wrappers named ``name`` call, with each wrapper's ``implementation`` cell as holder.

    With ``package=None`` the row covers a wrapper running transformers' Python
    reference (no library kernel is installed). With a package name it covers a
    wrapper running that library's kernel. A wrapper running any other library's
    kernel is located by no row and serves no sites.
    """

    def locate(context: Modeling) -> Located:
        for cells in context.wrappers:
            if cells["func_name"].cell_contents != name:
                continue
            implementation = unshimmed(cells["implementation"].cell_contents)
            reference = implementation is cells["torch_function"].cell_contents
            if reference if package is None else not reference and cells["package"].cell_contents == package:
                yield implementation, (Attr(cells["implementation"], "cell_contents"),)

    return locate


#: Every kernel served on a transformers model. Recurrent rows are named by their hub-kernel name.
CATALOG: list[Spec] = [
    Spec("eager_attention_forward", "attention", eager_functions, eager.serve_eager),
    Spec("grouped_mm_experts_forward", "experts", experts_forward("grouped_mm_experts_forward", "grouped_mm"), experts.serve_grouped_mm),
    Spec("batched_mm_experts_forward", "experts", experts_forward("batched_mm_experts_forward", "batched_mm"), experts.serve_batched_mm),
    Spec("chunk_gated_delta_rule", "recurrent", hub_kernel("chunk_gated_delta_rule"), recurrent.serve_gated_delta_chunk),
    Spec("fused_recurrent_gated_delta_rule", "recurrent", hub_kernel("fused_recurrent_gated_delta_rule"), recurrent.serve_gated_delta_recurrent),
    Spec("chunk_kda", "recurrent", hub_kernel("chunk_kda"), recurrent.serve_kda_chunk),
    Spec("fused_recurrent_kda", "recurrent", hub_kernel("fused_recurrent_kda"), recurrent.serve_gated_delta_recurrent),
    Spec("mamba_chunk_scan_combined", "recurrent", hub_kernel("mamba_chunk_scan_combined"), recurrent.serve_mamba2_chunk),
    Spec("selective_scan_fn", "recurrent", hub_kernel("selective_scan_fn"), recurrent.serve_mamba1_scan),
    Spec("selective_state_update", "recurrent", hub_kernel("selective_state_update"), recurrent.serve_mamba2_decode),
    Spec("fla.chunk_gated_delta_rule", "recurrent", hub_kernel("chunk_gated_delta_rule", "fla"), recurrent.serve_fla_gated_delta_chunk),
    Spec("fla.fused_recurrent_gated_delta_rule", "recurrent", hub_kernel("fused_recurrent_gated_delta_rule", "fla"), recurrent.serve_fla_gated_delta_recurrent),
]


def register(root: torch.nn.Module) -> None:
    """Register every catalog kernel that ``root``'s modules can reach."""
    register_catalog(CATALOG, modeling(root))
