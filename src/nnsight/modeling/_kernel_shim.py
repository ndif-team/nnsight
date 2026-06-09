"""Meta-load shim for CUDA-only kernel packages.

Some HuggingFace models ship custom remote modeling code (``trust_remote_code``)
that *hard-imports* CUDA/Triton kernel packages at module top level — e.g.
``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`` does::

    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn   # raises on CPU

Those packages (``mamba_ssm``, ``causal_conv1d``) cannot even be *installed*
without a CUDA toolchain, so a GPU-less nnsight/NDIF client cannot import the
model class — which means it cannot build the meta module tree it needs to
compile an intervention graph, and therefore cannot use ``remote=True`` at all.

On a meta-only client the model never runs a forward pass (the real forward
happens on the GPU host), so these kernels are never *called* — they are only
*imported*. This module installs inert stand-ins for the kernel packages around
the meta ``from_config`` call so the import resolves; every stand-in member
raises if it is ever actually invoked, so a dispatched/real run can never
silently use a stub instead of the real kernel.

The shim is a **no-op when the real kernel is installed** (e.g. on a GPU host),
so it never shadows a genuine implementation.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from typing import Dict, List

# top-level package -> {fully-qualified submodule it exposes: [members to stub]}.
# Only the members that remote modeling files import *unconditionally* strictly
# need stubbing; the rest are harmless and make the shim robust across the
# several Mamba-family remote files that copy this import block.
_KERNEL_STUBS: Dict[str, Dict[str, List[str]]] = {
    "mamba_ssm": {
        "mamba_ssm.ops.triton.layernorm_gated": ["rmsnorm_fn"],
        "mamba_ssm.ops.triton.selective_state_update": ["selective_state_update"],
        "mamba_ssm.ops.triton.ssd_combined": [
            "mamba_chunk_scan_combined",
            "mamba_split_conv1d_scan_combined",
        ],
    },
    "causal_conv1d": {
        "causal_conv1d": ["causal_conv1d_fn", "causal_conv1d_update"],
    },
}

# Set NNSIGHT_FORCE_META_KERNEL_SHIM=1 to install the stubs even when the real
# kernels are importable. Used to exercise the GPU-less client path on a machine
# that happens to have the kernels installed; not needed in normal operation.
_FORCE_ENV = "NNSIGHT_FORCE_META_KERNEL_SHIM"


def _make_stub_member(qualified_name: str):
    def _stub(*args, **kwargs):
        raise RuntimeError(
            f"{qualified_name} is an nnsight meta-load stub standing in for a "
            f"CUDA kernel that is not installed. It must not be called: build the "
            f"model on a meta-only client (no forward), or dispatch it on a CUDA "
            f"host with the real kernel package installed to actually run it."
        )

    _stub.__name__ = qualified_name.rsplit(".", 1)[-1]
    return _stub


def _register_module(name: str, is_package: bool, members: List[str], added: List[str]) -> types.ModuleType:
    module = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, loader=None)
    if is_package:
        spec.submodule_search_locations = []
        module.__path__ = []  # marks it importable as a package
    module.__spec__ = spec
    for member in members:
        setattr(module, member, _make_stub_member(f"{name}.{member}"))
    sys.modules[name] = module
    added.append(name)
    return module


@contextlib.contextmanager
def meta_kernel_shim(force: bool | None = None):
    """Temporarily satisfy CUDA-only kernel imports for meta construction.

    Installs lightweight stand-ins for any package in :data:`_KERNEL_STUBS` that
    is not already importable, then removes them on exit. A no-op for packages
    that are genuinely installed (unless ``force`` / the force env var is set).
    """

    if force is None:
        force = os.environ.get(_FORCE_ENV) == "1"

    added: List[str] = []
    try:
        for top, submodules in _KERNEL_STUBS.items():
            if top in sys.modules:
                continue  # already present (real or previously stubbed)
            if not force and importlib.util.find_spec(top) is not None:
                continue  # real kernel installed -> use it, don't shadow

            # create every dotted prefix as a package, then the leaf submodules
            packages_needed = set()
            for full in submodules:
                parts = full.split(".")
                for i in range(1, len(parts)):
                    packages_needed.add(".".join(parts[:i]))
            for pkg in sorted(packages_needed, key=lambda s: s.count(".")):
                if pkg not in sys.modules:
                    # a pure-package prefix carries no members of its own unless
                    # it is also a declared leaf (handled below)
                    _register_module(pkg, is_package=True, members=[], added=added)
            for full, members in submodules.items():
                is_pkg = full in packages_needed  # leaf that is also a package prefix
                if full in sys.modules:
                    # already created as a bare package prefix; attach members
                    for member in members:
                        setattr(sys.modules[full], member, _make_stub_member(f"{full}.{member}"))
                else:
                    _register_module(full, is_package=is_pkg, members=members, added=added)

            # wire parent.child attributes so ``from a.b.c import x`` resolves
            for name in list(sys.modules):
                if name == top or name.startswith(top + "."):
                    if "." in name:
                        parent, child = name.rsplit(".", 1)
                        if parent in sys.modules:
                            setattr(sys.modules[parent], child, sys.modules[name])
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)
