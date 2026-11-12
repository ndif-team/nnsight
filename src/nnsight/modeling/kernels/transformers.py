"""Catalog and locators for the kernels called by transformers models.

`CATALOG` lists the supported kernels as [`Spec`][nnsight.intervention.kernel.Spec]
rows, each pairing a site family and serve function with a locator that yields
matching functions and their holders. `register_model` runs `CATALOG` against a
model when a `TransformersModel` is constructed. Because the kernel registry is
process-wide, a function registered for one model serves every model that reads
its registered holders.

Models in transformers reach their kernels in four ways, each handled by a
locator:

* `eager_functions`: under ``attn_implementation="eager"``, an attention module
  passes its modeling module's own ``*eager_attention_forward`` global as the
  default to ``ALL_ATTENTION_FUNCTIONS.get_interface`` on each call.
* `experts_forward`: the experts wrapper looks up its implementation in
  ``ALL_EXPERTS_FUNCTIONS`` on each call.
* `flash_function`: under ``attn_implementation="flash_attention_2"``,
  ``_flash_attention_forward`` reads the dense and varlen FlashAttention
  functions from two module globals on each call.
* `hub_kernel`: recurrent kernels are called through
  ``use_kernel_func_from_hub_with_fallback`` wrappers, which invoke the callable
  stored in their ``implementation`` closure cell (transformers' Python
  reference or an installed library's kernel).

Site readers live in the sibling modules eager.py, flash.py, recurrent.py, and experts.py.
PyTorch's SDPA has its own single-row catalog in intervention/attention.py.

The hub-kernel wrappers exist from transformers 4.50 and carry the closure
variables the locator reads from 5.17. On an older release `modeling` finds no
wrappers and the recurrent rows locate nothing, so the model is built without
recurrent sites.
"""

from __future__ import annotations

import sys
from types import CellType, FunctionType, ModuleType
from typing import Callable, NamedTuple

import torch

from ...intervention.kernel import Attr, Item, Located, Spec, register_catalog, unshimmed
from . import eager, experts, flash, recurrent

#: The closure variables of a hub-kernel wrapper that `hub_kernel` reads.
_WRAPPER_CELLS = frozenset({"func_name", "package", "torch_function", "implementation"})


class Modeling(NamedTuple):
    """Modeling modules and hub-kernel wrapper closures inspected by catalog locators for one model."""

    #: The Python modules that define the classes of the model's submodules.
    modules: list[ModuleType]
    #: One dict per hub-kernel wrapper in those modules, mapping each closure variable name to its cell.
    wrappers: list[dict[str, CellType]]


def modeling(root: torch.nn.Module) -> Modeling:
    """Collect the modeling modules of ``root``'s submodules and the hub-kernel wrappers they define."""
    # A dynamically exec'd class has no entry in `sys.modules`.
    modules = list({sys.modules.get(type(sub).__module__) for sub in root.modules()} - {None})
    try:
        from transformers.integrations import hub_kernels  # noqa: F401
    except ImportError:  # transformers before 4.50 has no hub-kernel wrappers
        return Modeling(modules, [])
    return Modeling(modules, [cells for module in modules for cells in _hub_wrappers(module)])


def _hub_wrappers(module: ModuleType) -> list[dict[str, CellType]]:
    """Inspect all ``use_kernel_func_from_hub_with_fallback`` wrapper closures defined in ``module``.

    Each wrapper is identified by its code object's name (``"wrapped"``) and file
    (``transformers.integrations.hub_kernels``). Each entry in the returned list
    maps a closure variable name (``func_name``, ``package``, ``torch_function``,
    ``implementation``, etc.) to its closure cell, allowing callers to read
    ``cell_contents`` or pass the cell as a holder. The wrappers of transformers
    5.15 and 5.16 do not close over ``func_name`` and ``package``, so they cannot
    be matched to a catalog row and are left out.
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
        cells = dict(zip(value.__code__.co_freevars, value.__closure__))
        if _WRAPPER_CELLS <= cells.keys():
            found.append(cells)
    return found


def eager_functions(context: Modeling) -> Located:
    """Locate every eager attention function defined in the model's modeling modules.

    No explicit holder is yielded: [`register`][nnsight.intervention.kernel.register]
    adds the function's own module global automatically, which is the binding the
    attention module passes on each call.
    """
    for module in context.modules:
        for value in list(vars(module).values()):
            # If a trace is open while this model is constructed, module globals already hold shims.
            fn = unshimmed(value)
            if eager.is_eager_forward(fn):
                yield fn, ()


def experts_forward(name: str, key: str) -> Callable[[Modeling], Located]:
    """Locate the experts forward ``name``, held at ``key`` of transformers' experts-implementation mapping.

    ``ALL_EXPERTS_FUNCTIONS.__setitem__`` writes a local override entry, so the
    holder is the underlying entry in ``_global_mapping``. A transformers
    release without ``transformers.integrations.moe`` locates nothing.
    """

    def locate(context: Modeling) -> Located:
        try:
            from transformers.integrations import moe
        except ImportError:
            return
        yield unshimmed(vars(moe)[name]), (Item(moe.ALL_EXPERTS_FUNCTIONS._global_mapping, key),)

    return locate


def hub_kernel(name: str, package: str | None = None) -> Callable[[Modeling], Located]:
    """Locate the callable invoked by hub-kernel wrappers named ``name``, with each wrapper's ``implementation`` cell as holder.

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
            wanted = reference if package is None else (not reference and cells["package"].cell_contents == package)
            if wanted:
                yield implementation, (Attr(cells["implementation"], "cell_contents"),)

    return locate


def flash_function(name: str) -> Callable[[Modeling], Located]:
    """Locate the FlashAttention function transformers holds in its module global ``name``.

    transformers fills ``_flash_fn`` and ``_flash_varlen_fn`` when a model is
    loaded with a flash attention implementation and reads them on every call.
    The global is ``None`` when no model asked for flash attention, and absent on
    releases that imported the functions under their own names.
    """

    def locate(context: Modeling) -> Located:
        try:
            from transformers import modeling_flash_attention_utils as utils
        except ImportError:
            return
        fn = unshimmed(vars(utils).get(name))
        if fn is not None:
            yield fn, (Attr(utils, name),)

    return locate


#: Supported kernels on transformers models. Recurrent rows are labeled by their hub-kernel name (prefixed with
#: ``"{package}."`` for library kernels); the function a Python reference row registers has a different name.
CATALOG: list[Spec] = [
    Spec("eager_attention_forward", "attention", eager_functions, eager.serve_eager),
    Spec("flash_attn_func", "attention", flash_function("_flash_fn"), flash.serve),
    Spec("flash_attn_varlen_func", "attention", flash_function("_flash_varlen_fn"), flash.serve),
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


def register_model(root: torch.nn.Module) -> None:
    """Register every function `CATALOG` locates for ``root``.

    The experts rows locate their forwards for every model; the other rows locate
    only what ``root``'s modeling modules define.
    """
    register_catalog(CATALOG, modeling(root))
