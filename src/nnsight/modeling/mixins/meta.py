"""Building a model on the meta device, dispatching real weights only on demand.

Loading a large model's weights is slow and memory-hungry, but most of what
nnsight needs to plan a trace — the module tree and the shapes that flow through
it — is fixed by the model's *structure*, not its weights. So a model is built
lazily: every module is constructed on the *meta* device, which records a
tensor's shape and dtype but allocates no storage. The envoy tree, `scan`,
and remote-key resolution all work against this weightless skeleton.

[`MetaDevice`][nnsight.modeling.mixins.meta.MetaDevice] is the mechanism — a torch function mode that forces every
tensor created (or moved) within it onto meta, however it was created.
[`Meta`][nnsight.modeling.mixins.meta.Meta] is the policy — it does the meta build up front and *dispatches*
(loads real weights and swaps them into the same envoy tree) the first time the
model actually has to run.

.. code-block:: python

    model = TransformersModel("openai-community/gpt2")  # meta build, no weights
    model.scan("Hello")           # inspect shapes, still no weights
    with model.trace("Hello"):    # dispatches real weights on first run
        ...

The meta build reconstructs structure only, so a subclass's `Meta._load_meta`
drops weight- and placement-related kwargs (``device_map``, ``max_memory``, ...):
they mean nothing on storageless meta tensors and take effect only at dispatch.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Iterator

import torch
from torch._guards import detect_fake_mode
from torch.overrides import TorchFunctionMode

from ...intervention.envoy import Envoy
from ...intervention.tracer import ScanningTracer
from .loadable import Loadable

# Legacy ``Tensor.type("torch.cuda.FloatTensor")`` strings, keyed by their
# trailing class name, mapped to the dtype they denote (the device is dropped).
_LEGACY_TENSOR_DTYPES = {
    "FloatTensor": torch.float32,
    "DoubleTensor": torch.float64,
    "HalfTensor": torch.float16,
    "BFloat16Tensor": torch.bfloat16,
    "ByteTensor": torch.uint8,
    "CharTensor": torch.int8,
    "ShortTensor": torch.int16,
    "IntTensor": torch.int32,
    "LongTensor": torch.int64,
    "BoolTensor": torch.bool,
}


class MetaDevice(TorchFunctionMode):
    """Force every tensor created within the context onto the meta device.

    Setting a default device only covers factory calls that omit ``device=``;
    this also rewrites explicit ``device=`` arguments, so tensors land on meta
    no matter how they are created.

    [`real`][nnsight.modeling.mixins.meta.MetaDevice.real] opens a nested window where this forcing is suspended, for the
    parts of a lazy build that need genuine tensors (e.g. a component whose
    constructor moves a buffer with ``.to(...)``, which a meta tensor can't do).
    """

    _META = torch.device("meta")
    _suspended = threading.local()

    def __enter__(self) -> MetaDevice:
        # Force torch's lazy machinery to initialize on a real device first;
        # e.g. importing torch._dynamo creates tensors and calls .item(), which
        # would crash if those tensors were forced onto meta.
        import torch._dynamo  # noqa: F401

        self._default = torch.device("meta")
        self._default.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        super().__exit__(*exc)
        self._default.__exit__(*exc)

    def _meta_target(self, arg: Any) -> Any:
        # Map a ``.to`` positional to what keeps the tensor on meta: a device
        # (object/string/ordinal) becomes meta; another tensor contributes only
        # its dtype (its device is dropped); dtypes and the ``non_blocking`` /
        # ``copy`` bool flags pass through untouched.
        if isinstance(arg, (torch.device, str)):
            return self._META
        if isinstance(arg, bool):
            return arg
        if isinstance(arg, int):
            return self._META
        if isinstance(arg, torch.Tensor):
            return arg.dtype
        return arg

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if getattr(self._suspended, "on", False):
            return func(*args, **kwargs)
        # Redirect every destination device to meta so tensors always land
        # there, however they're created or moved. Moving a meta tensor onto a
        # real device would otherwise raise — there's no data to copy — so these
        # become no-ops on meta, letting a constructor that relocates a buffer
        # during init just work. Value-dependent ops (``.item()``, ``.numpy()``,
        # data-dependent branching) can't be faked this way; build those
        # components under [`real`][nnsight.modeling.mixins.meta.MetaDevice.real] instead.
        if kwargs.get("device") is not None:
            kwargs["device"] = self._META
        if func is torch.Tensor.to:
            # Covers .to("cpu"), .to(device=...), .to(0), .to(other_tensor), and
            # dtype-only .to(float16) (which has no destination and is untouched).
            args = (args[0], *(self._meta_target(arg) for arg in args[1:]))
        elif func in (torch.Tensor.cpu, torch.Tensor.cuda, torch.Tensor.pin_memory):
            return args[0].to(self._META)
        elif func is torch.Tensor.type_as:
            return args[0].to(self._META, args[1].dtype)
        elif func is torch.Tensor.type and len(args) > 1 and isinstance(args[1], str):
            return args[0].to(self._META, _LEGACY_TENSOR_DTYPES[args[1].rsplit(".", 1)[-1]])
        return func(*args, **kwargs)

    @classmethod
    @contextlib.contextmanager
    def real(cls) -> Iterator[None]:
        """Suspend meta-forcing within this block so tensors are created for real."""
        previous = getattr(cls._suspended, "on", False)
        cls._suspended.on = True
        try:
            # Also override the meta default device for factory calls with no
            # explicit device, which our __torch_function__ does not redirect.
            with torch.device("cpu"):
                yield
        finally:
            cls._suspended.on = previous


class Meta(Loadable):
    """A model wrapper that builds on meta and dispatches real weights on demand.

    Extends [`Loadable`][nnsight.modeling.mixins.loadable.Loadable] with a two-phase
    build: `_load_meta` constructs the weightless skeleton up front, and
    [`dispatch`][nnsight.modeling.mixins.meta.Meta.dispatch] (triggered on the first `interleave` that isn't a scan)
    loads real weights via ``_load`` and swaps them into the existing envoy tree.
    Passing a ready module, or ``dispatch=True``, skips the meta phase and loads
    eagerly.
    """

    def __init__(
        self,
        *args: Any,
        dispatch: bool = False,
        rename: Any = None,
        envoys: Any = None,
        **kwargs: Any,
    ) -> None:
        self.dispatched = False

        # rename/envoys thread to the Envoy but stay out of self.kwargs, so they
        # aren't replayed into _load on dispatch; aliases survive dispatch anyway
        # because _update re-points the same envoy objects the aliases reference.
        if (args and isinstance(args[0], torch.nn.Module)) or dispatch:
            self.dispatched = True
            super().__init__(*args, rename=rename, envoys=envoys, **kwargs)
        else:
            with MetaDevice():
                model = self._load_meta(*args, **kwargs)
            Envoy.__init__(self, model, rename=rename, envoys=envoys)

        self.args = args
        self.kwargs = kwargs

    def _load_meta(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        """Build the model's structure on the meta device (no weights loaded).

        Base default: not implemented. Subclasses (e.g.
        [`TransformersModel`][nnsight.modeling.transformers.TransformersModel]) build the module
        tree from config so shapes are known without any weights in memory.
        """
        raise NotImplementedError()

    def dispatch(self) -> None:
        """Load real weights via ``_load`` and swap them into the envoy tree.

        Idempotent — a no-op once dispatched. Triggered automatically on the first
        real (non-scan) `interleave`; call it directly to load eagerly.
        """
        if self.dispatched:
            return
        model = self._load(*self.args, **self.kwargs)
        self._update(model)
        self.dispatched = True

    def scan(
        self, *args: Any, fn: Any = None, backend: Any = None, **kwargs: Any
    ) -> ScanningTracer:
        """Inspect activation shapes without loading real weights or computing.

        Just like [`trace`][nnsight.intervention.envoy.Envoy.trace], but the forward
        runs under a fake-tensor mode so only tensor metadata (shapes, dtypes)
        propagates — the model is never dispatched, so this works on a
        meta-initialized model with no weights in memory:

            >>> model = TransformersModel("openai-community/gpt2")  # not dispatched
            >>> with model.scan("Hello World"):
            ...     hidden = model.transformer.h[0].output[0].shape
            >>> print(hidden)  # torch.Size([...])

        The values read inside the block are fake tensors valid only within it;
        read their ``.shape`` / ``.dtype`` here rather than saving the tensor out.

        Args:
            *args: Inputs to scan, forwarded to the model's forward pass.
            fn: Method to run, resolved against the module. Defaults to ``__call__``.
            backend: Backend that runs the captured block. Defaults to in place.
            **kwargs: Keyword inputs forwarded to the forward pass.

        Returns:
            A [`ScanningTracer`][nnsight.intervention.tracer.ScanningTracer] for this model.
        """
        if fn is None:
            fn = "__call__"
        return ScanningTracer(self, fn, *args, backend=backend, **kwargs)

    def interleave(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        # Interleaving an undispatched model needs real weights, so load them
        # first. fn is resolved against the live module in super().interleave,
        # so a named fn binds to the freshly loaded one with no rebinding.
        #
        # Exception: a scan runs the forward under a fake-tensor mode purely to
        # propagate shapes, so real weights aren't needed. Detecting the active
        # fake mode (rather than the scanning tracer type) keeps this mixin
        # decoupled from the intervention layer; the meta parameters then take
        # part in the fake forward directly.
        if not self.dispatched and detect_fake_mode() is None:
            self.dispatch()

        return super().interleave(fn, *args, **kwargs)


    def __setstate__(self, state):
        super().__setstate__(state)
        # A deserialized model resolves to the server's already-loaded module, so
        # it is dispatched — never re-run the lazy meta build for it.
        self.dispatched = True