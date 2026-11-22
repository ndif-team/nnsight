"""Single-device view for intervention values on a multi-device model.

When a model's parameters span multiple devices, each module computes on its own
device and produces tensors there. Without device movement at the intervention
boundary, combining tensors taken from modules on different devices (such as
concatenating two inputs of a layer or adding a row of an output projection
weight to an earlier hidden state) raises ``RuntimeError: Expected all tensors
to be on the same device``.

When a model has parameters on two or more devices, `Devices` places tensors
exposed to intervention code onto the device of the model's first parameter
and applies changes back to the original tensors when a location visit ends:

* **Assigned a replacement tensor** (``layer.output = x``): the replacement is
  moved to the device of the original tensor at that position (or the device of
  the first moved tensor if the container structure changed).
* **Edited in place** (``hidden += d``, ``hidden[:] = x``): the edit modifies
  the copy on the serving device and increments its ``_version`` counter. When
  the visit ends, `restore` copies the modified data back into the original
  tensor with ``copy_``, so later operations in the forward pass that reference
  the original tensor read the updated values. Unmodified copies are not copied
  back.
* **Marked with ``requires_grad_(True)``** on a copy whose original tensor did
  not require grad: at an ``.output`` location the copy is moved to the
  original tensor's device and replaces it; at any other location the copy is
  written into the original tensor with ``copy_`` so the original tensor stays
  connected to the copy in autograd.

When the original tensor already requires grad, `_copy` records the mapping
from the copy to the original tensor (`origin`). A read leaves the original
tensor in place for the forward pass, and a backward trace registers its
gradient hook on the original tensor and moves the gradient to the copy's
device.

In-place edits to a moved copy are applied only when the visit to that location
ends; edits made after the forward pass has moved past that location are not
copied back.

Module parameter and buffer attributes accessed on an `Envoy` during a run are
copied to the serving device by `weight`, and `settle` writes in-place edits on
those copies back to the original tensors. Ad-hoc module calls move their inputs
to the target module's device (`call`) and move their outputs to the serving
device (`result`). Tensors in a location's value that share storage with a model
parameter or buffer are left on their own device.

In-place writes made through ``tensor.data`` do not increment ``_version`` and
are not detected on moved copies.

`Interleaver.handle` applies `serve` and `restore` once per location visit,
around the worker loop and cache observers.
"""

from __future__ import annotations

import copy
import itertools
import weakref
from typing import Any, Callable, Optional

import torch


def _version(tensor: torch.Tensor) -> Optional[int]:
    """Return ``tensor._version``, or ``None`` if accessing it raises
    ``RuntimeError`` (as on an inference tensor)."""
    try:
        return tensor._version
    except RuntimeError:
        return None


# Map id(copy) -> original tensor for moved copies participating in autograd.
# Entries are removed by weakref.finalize when the copy is garbage-collected.
_origins: dict[int, torch.Tensor] = {}


def origin(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the original model tensor behind a moved copy ``tensor`` if it was
    bound for autograd, or ``None`` otherwise.

    Used by `BackwardTracer` to attach gradient hooks to the original tensor on
    its compute device while serving gradients on the copy's device.
    """
    return _origins.get(id(tensor))


def _copy(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy ``tensor`` to ``device`` with a version counter enabled."""
    if torch.is_inference_mode_enabled():
        # Tensors created under inference_mode do not track _version. Disable
        # inference_mode and grad while copying so the copy tracks in-place edits.
        with torch.inference_mode(False), torch.no_grad():
            return tensor.to(device)
    copy = tensor.to(device)
    if copy.requires_grad:
        _bind(copy, tensor)
    return copy


def _bind(copy: torch.Tensor, tensor: torch.Tensor) -> None:
    """Register ``tensor`` as the original tensor for ``copy`` in `_origins` (see `origin`)."""
    _origins[id(copy)] = tensor
    weakref.finalize(copy, _origins.pop, id(copy), None)


def _map(value: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    """Return ``value`` with ``fn`` applied to every tensor inside its tuples,
    named tuples, lists, dicts, and dict subclasses.

    Other objects are returned unchanged.
    """
    if isinstance(value, torch.Tensor):
        return fn(value)
    if type(value) is tuple:
        return tuple(_map(item, fn) for item in value)
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # a named tuple
        return type(value)(*(_map(item, fn) for item in value))
    if type(value) is list:
        return [_map(item, fn) for item in value]
    if type(value) is dict:
        return {key: _map(item, fn) for key, item in value.items()}
    if isinstance(value, dict):  # a dict subclass, e.g. a transformers ModelOutput
        mapped = copy.copy(value)
        for key, item in value.items():
            mapped[key] = _map(item, fn)
        return mapped
    return value


def _pairs(before: Any, after: Any, out: list) -> bool:
    """Append ``(before, after)`` tensor pairs at matching positions to ``out``;
    return whether ``before`` and ``after`` have the same container types,
    lengths, keys, and leaf types.
    """
    if isinstance(before, torch.Tensor) or isinstance(after, torch.Tensor):
        out.append((before, after))
        return isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor)
    if type(before) is not type(after):
        return before is after
    if isinstance(before, (tuple, list)):
        return len(before) == len(after) and all(
            _pairs(a, b, out) for a, b in zip(before, after)
        )
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(
            _pairs(before[key], after[key], out) for key in before
        )
    return True


class Devices:
    """Single-device view for a run on a model whose parameters span multiple devices.

    Created in [`Interleaver.__enter__`][nnsight.intervention.interleaver.Interleaver.__enter__]
    when the root model has parameters on two or more devices and cleared when
    the run exits. When all parameters are on one device, ``interleaver.devices``
    is ``None``.
    """

    def __init__(self, device: torch.device, owned: set[int]) -> None:
        self.device = device
        # Storage data pointers of the model's non-meta parameters and buffers.
        # Tensors sharing one of these storages are not copied by `serve`.
        self.owned = owned
        # Parameter and buffer copies on `device` for this run: id(tensor) ->
        # [tensor, copy on `device`, copy._version when last settled,
        # _version(tensor) when last copied].
        self.weights: dict[int, list] = {}

    @classmethod
    def of(cls, envoy: Any) -> "Optional[Devices]":
        """Return a `Devices` instance targeting ``envoy.device`` when ``envoy``
        has parameters on two or more devices, or ``None`` otherwise."""
        device = envoy.device
        if device is None or len(envoy.devices) < 2:
            return None
        module = envoy._module
        owned = {
            tensor.untyped_storage().data_ptr()
            for tensor in itertools.chain(module.parameters(), module.buffers())
            if tensor.device.type != "meta"
        }
        return cls(device, owned)

    def serve(self, location: str, value: Any) -> "tuple[Any, Optional[Callable[[Any], Any]]]":
        """Copy eligible tensors in ``value`` to `device` and return ``(served, restore)``.

        ``restore`` maps the post-intervention value back to the original
        tensors and devices for the forward pass. ``restore`` is ``None`` when
        no tensor in ``value`` was copied (every tensor was already on `device`,
        on ``meta``, or shared storage with `owned`).
        """
        device = self.device
        moved: dict[int, torch.Tensor] = {}  # id(original) -> its copy
        homes: dict[int, tuple] = {}  # id(copy) -> (original, copy, version when served)
        kept: set[int] = set()  # ids of the tensors served without copying

        def move(tensor: torch.Tensor) -> torch.Tensor:
            twin = moved.get(id(tensor))
            if twin is not None:  # reuse the same copy when a tensor appears twice
                return twin
            if (
                tensor.device == device
                or tensor.device.type == "meta"
                or tensor.untyped_storage().data_ptr() in self.owned
            ):
                kept.add(id(tensor))
                return tensor
            twin = _copy(tensor, device)
            moved[id(tensor)] = twin
            homes[id(twin)] = (tensor, twin, twin._version)
            return twin

        served = _map(value, move)
        if not moved:
            return value, None

        def restore(returned: Any) -> Any:
            # Propagate in-place edits and new requires_grad flags from every
            # moved copy back to its original tensor, even if the copy was
            # later replaced in `returned` by an assignment.
            settled: dict[int, torch.Tensor] = {}
            for original, twin, version in homes.values():
                if twin.requires_grad and not original.requires_grad:
                    if location.endswith(".output"):
                        settled[id(twin)] = twin.to(original.device)
                        continue
                    original.copy_(twin)
                elif twin._version != version:
                    original.copy_(twin)
                settled[id(twin)] = original

            # Move any newly assigned tensor to the device of the original
            # tensor at its matching position, or to the device of the first
            # moved tensor when the structure of `returned` differs from `value`.
            pairs: list = []
            expected = (
                {
                    id(back): original.device
                    for original, back in pairs
                    if isinstance(back, torch.Tensor)
                }
                if _pairs(value, returned, pairs)
                else {}
            )
            fallback = next(iter(homes.values()))[0].device

            def home(tensor: torch.Tensor) -> torch.Tensor:
                known = settled.get(id(tensor))
                if known is not None:
                    return known
                if id(tensor) in kept or tensor.device.type == "meta":
                    return tensor
                return tensor.to(expected.get(id(tensor), fallback))

            return _map(returned, home)

        return served, restore

    def weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a module parameter or buffer on `device`.

        Tensors already on `device` or on ``meta`` are returned unchanged.
        Tensors on other devices are copied once per run into `weights`, and
        refreshed with ``copy_`` if ``_version(tensor)`` has changed since the
        copy was made or settled.

        Inside a backward trace, reading ``.grad`` on the copy reads the
        gradient of the original tensor via `origin`; after the trace, the
        accumulated ``.grad`` attribute resides on the original parameter.
        """
        if tensor.device == self.device or tensor.device.type == "meta":
            return tensor
        entry = self.weights.get(id(tensor))
        if entry is None:
            copy = _copy(tensor, self.device)
            entry = self.weights[id(tensor)] = [tensor, copy, copy._version, _version(tensor)]
        elif _version(tensor) != entry[3]:
            with torch.no_grad():
                entry[1].copy_(tensor)
            entry[2], entry[3] = entry[1]._version, _version(tensor)
        return entry[1]

    def call(self, module: torch.nn.Module, inputs: Any) -> Any:
        """Settle edited weight copies and move tensors in ``inputs`` to the
        device of ``module``'s first parameter (when ``module`` has a non-meta
        parameter).
        """
        self.settle()
        parameter = next(module.parameters(), None)
        if parameter is None or parameter.device.type == "meta":
            return inputs
        return _map(inputs, lambda tensor: tensor.to(parameter.device))

    def result(self, value: Any) -> Any:
        """Move tensors in ``value`` (an ad-hoc module call's output) to `device`."""
        return _map(value, lambda tensor: tensor.to(self.device))

    def skipped(self, value: Any, inputs: Any) -> Any:
        """Move tensors in a ``.skip`` replacement ``value`` to the device of
        the first tensor in ``inputs`` (or return ``value`` unchanged if
        ``inputs`` contains no tensors).
        """
        found: list = []
        _map(inputs, found.append)
        if not found:
            return value
        return _map(value, lambda tensor: tensor.to(found[0].device))

    def settle(self) -> None:
        """Propagate ``requires_grad`` and in-place edits from cached weight
        copies back to the model's original parameter and buffer tensors.

        Called after workers run in `Interleaver.__enter__`, `Interleaver.handle`,
        and before ad-hoc module calls in `call`.
        """
        for entry in self.weights.values():
            weight, copy = entry[0], entry[1]
            if copy.requires_grad and not weight.requires_grad:
                # If the block enabled requires_grad on a copy of a frozen
                # weight, enable it on the original weight and bind the copy in
                # `_origins` so backward hooks can resolve it.
                weight.requires_grad_(True)
                _bind(copy, weight)
            if copy._version != entry[2]:
                # Copy under no_grad so leaf parameters that require grad can
                # be updated in place.
                with torch.no_grad():
                    weight.copy_(copy)
                entry[2], entry[3] = copy._version, _version(weight)
