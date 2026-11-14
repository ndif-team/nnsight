"""Catalog and locators for the kernels called by transformers models.

`CATALOG` lists the supported kernels as [`Spec`][nnsight.intervention.kernel.Spec]
rows, each pairing a site family and serve function with a locator that yields
matching functions and their holders. `register_model` runs `CATALOG` against a
model when a `TransformersModel` is constructed. The FlashAttention 2 rows,
`FLASH`, run at the start of every trace instead (`register_flash`), because
the functions transformers holds for them change after construction. Because
the kernel registry is process-wide, a function registered for one model serves
every model that reads its registered holders.

Models in transformers reach their kernels in four ways, each handled by a
locator:

* `eager_functions`: under ``attn_implementation="eager"``, an attention module
  passes its modeling module's own ``*eager_attention_forward`` global as the
  default to ``ALL_ATTENTION_FUNCTIONS.get_interface`` on each call.
* `experts_forward`: the experts wrapper looks up its implementation in
  ``ALL_EXPERTS_FUNCTIONS`` on each call.
* `flash_function`: under ``attn_implementation="flash_attention_2"``,
  ``_flash_attention_forward`` reads the dense and varlen FlashAttention 2
  functions from two module globals on each call.
* `hub_kernel`: recurrent kernels are called through
  ``use_kernel_func_from_hub_with_fallback`` wrappers, which invoke the callable
  stored in their ``implementation`` closure cell (transformers' Python
  reference or an installed library's kernel). With the ``kernels`` package
  installed, transformers hands each wrapper to ``kernels``, which returns a
  module whose ``forward`` calls it; `_hub_wrappers` reads the wrapper inside.

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
from typing import Any, Callable, NamedTuple

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
    (``transformers.integrations.hub_kernels``). With the ``kernels`` package
    installed, the module global holds a ``kernels`` module instance whose
    ``forward`` closes over the wrapper as ``func``, and the wrapper is read from
    that cell. Unless the model is loaded with ``use_kernels=True``, which
    replaces that ``forward`` with a hub kernel, the wrapper still runs on every
    call. Each entry in the returned list maps a closure variable name
    (``func_name``, ``package``, ``torch_function``, ``implementation``, etc.) to
    its closure cell, allowing callers to read ``cell_contents`` or pass the cell
    as a holder. The wrappers of transformers 5.15 and 5.16 do not close over
    ``func_name`` and ``package``, so they cannot be matched to a catalog row and
    are left out.
    """
    from transformers.integrations import hub_kernels

    found = []
    for value in vars(module).values():
        if isinstance(value, torch.nn.Module) and isinstance(type(value).forward, FunctionType):
            forward = type(value).forward
            closure = dict(zip(forward.__code__.co_freevars, forward.__closure__ or ()))
            if "func" in closure:
                value = closure["func"].cell_contents
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
    """Locate the flash attention function transformers holds in its module global ``name``.

    transformers fills ``_flash_fn`` and ``_flash_varlen_fn`` with the functions
    of one flash attention implementation at a time and reads them on every
    call. The locator yields whatever the global holds, so it runs only from
    `register_flash`, after that has loaded a FlashAttention 2 implementation.
    """

    def locate(context: Modeling) -> Located:
        from transformers import modeling_flash_attention_utils as utils

        yield unshimmed(vars(utils)[name]), (Attr(utils, name),)

    return locate


#: Supported kernels on transformers models. Recurrent rows are labeled by their hub-kernel name (prefixed with
#: ``"{package}."`` for library kernels); the function a Python reference row registers has a different name.
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


#: The FlashAttention 2 rows, registered for each trace by `register_flash`.
FLASH: list[Spec] = [
    Spec("flash_attn_func", "attention", flash_function("_flash_fn"), flash.serve),
    Spec("flash_attn_varlen_func", "attention", flash_function("_flash_varlen_fn"), flash.serve),
]


def register_model(root: torch.nn.Module) -> None:
    """Register every function `CATALOG` locates for ``root``.

    The experts rows locate their forwards for every model; the other rows locate
    only what ``root``'s modeling modules define.
    """
    register_catalog(CATALOG, modeling(root))


def _implementations(config: Any) -> set[str]:
    """The attention implementations set on ``config`` and its sub-configs (a vision-language model sets one per tower)."""
    found = {config._attn_implementation}
    for name in config.sub_configs:
        sub = getattr(config, name)
        if sub is not None:
            found |= _implementations(sub)
    return found


def register_flash(root: torch.nn.Module) -> None:
    """Load ``root``'s FlashAttention 2 functions into transformers' globals and register them, before a trace of ``root``.

    transformers keeps the flash functions of the implementation that last ran
    in module globals and swaps them when a forward runs with another one, so
    what the globals hold when the model is constructed says nothing about the
    model's later traces: the model may be switched to FlashAttention 2 after
    construction (``set_attn_implementation``, or a remote request's
    ``attn_implementation``), or another model may load a different flash
    implementation in between. This runs before every trace's shims are
    installed. It loads the model's implementation the way the model's forward
    would (``lazy_import_flash_attention``), so the forward finds it loaded and
    keeps the patched globals, and registers the functions it leaves there. When
    a model with a different flash implementation runs inside the same open
    trace, its forward swaps the globals, and FlashAttention 2 calls after that
    are not served.
    """
    try:
        from transformers import PreTrainedModel
        from transformers import modeling_flash_attention_utils as utils
    except ImportError:
        return
    if "FLASH_ATTN_KERNEL_FALLBACK" not in vars(utils):  # releases before the hub-kernel fallback keep other globals
        return
    if not isinstance(root, PreTrainedModel):  # a wrapped plain module has no attention implementation
        return
    wanted = _implementations(root.config) & {"flash_attention_2", utils.FLASH_ATTN_KERNEL_FALLBACK["flash_attention_2"]}
    # A model whose towers mix the package and hub spellings cannot have both loaded at once, so it gets no sites.
    if len(wanted) != 1:
        return
    utils.lazy_import_flash_attention(wanted.pop())
    register_catalog(FLASH)
