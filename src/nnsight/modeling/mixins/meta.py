import threading
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights

from ...intervention.tracing.tracer import ScanningTracer
from .. import NNsight
from .loadable import LoadableMixin


class MetaMixin(LoadableMixin):
    """Mixin that adds lazy loading with meta tensors and deferred dispatch.

    When ``dispatch=False`` (the default), the model is initialized with
    meta tensors (no memory allocation) using :meth:`_load_meta`. Real
    weights are loaded later when :meth:`dispatch` is called, or
    automatically on the first :meth:`trace` / :meth:`generate` call.

    When ``dispatch=True`` or a pre-loaded ``torch.nn.Module`` is passed,
    the model is loaded immediately via :meth:`_load`.

    Args:
        *args: Forwarded to :meth:`_load_meta` or :meth:`_load`.
        dispatch (bool): If ``True``, load real weights immediately.
            Defaults to ``False`` (lazy / meta-tensor initialization).
        meta_buffers (bool): If ``True``, buffers are also created on
            the meta device. Defaults to ``True``.
        rename (Optional[Dict[str, str]]): Module path aliases.
        **kwargs: Forwarded to :meth:`_load_meta` or :meth:`_load`.

    Attributes:
        dispatched (bool): Whether real weights have been loaded.
        args: Saved positional arguments for deferred :meth:`dispatch`.
        kwargs: Saved keyword arguments for deferred :meth:`dispatch`.
    """

    def __init__(
        self,
        *args,
        dispatch: bool = False,
        meta_buffers: bool = True,
        rename: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:

        self.dispatched = False
        # Guards ``dispatch()`` so two concurrent callers can't both run
        # ``_load`` → ``_update`` and leave the Envoy tree in whichever
        # order the last ``_update`` happened to finish.
        self._dispatch_lock = threading.Lock()

        if isinstance(args[0], torch.nn.Module) or dispatch:

            self.dispatched = True

            super().__init__(*args, rename=rename, **kwargs)

        else:

            with init_empty_weights(include_buffers=meta_buffers):

                model = self._load_meta(*args, **kwargs)

            NNsight.__init__(self, model, rename=rename)

        self.args = args
        self.kwargs = kwargs

    def _load_meta(self, *args, **kwargs) -> torch.nn.Module:
        """Load a lightweight meta-tensor version of the model.

        Called when ``dispatch=False`` to build the model architecture
        without allocating real weight memory. Must be implemented by
        subclasses.

        Returns:
            torch.nn.Module: A model with meta-device parameters.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """

        raise NotImplementedError()

    def dispatch(self) -> None:
        """Load real weights into the model, replacing meta tensors.

        Idempotent and thread-safe. If two callers race into
        ``dispatch()`` concurrently, only the first loads and updates;
        the second sees ``dispatched=True`` under the lock and returns
        immediately. Prevents double-``_load`` / double-``_update``
        from leaving the Envoy tree in whichever order the last
        ``_update`` happened to finish — a hazard that would otherwise
        manifest as corrupted envoy bindings under concurrent serve
        traffic or interleaving auto-dispatch paths.

        Calls :meth:`_load` with the saved constructor arguments and
        updates the Envoy tree to reflect the newly loaded module.
        After this call, :attr:`dispatched` is ``True``.
        """
        # Fast path — no lock contention when the common case (already
        # dispatched) hits.
        if self.dispatched:
            return

        with self._dispatch_lock:
            # Double-checked under the lock: another caller may have
            # completed dispatch between our fast-path check and the
            # acquire.
            if self.dispatched:
                return

            model = self._load(*self.args, **self.kwargs)
            self._update(model)
            # TODO legacy
            self.__dict__["_model"] = self._module

            self.dispatched = True

    def interleave(self, fn: Callable, *args, **kwargs):
        """Run *fn* with interleaved interventions, auto-dispatching if needed.

        If the model has not been dispatched yet and this is not a scan,
        :meth:`dispatch` is called first and *fn* is rebound to the
        newly loaded module.
        """

        if not self.dispatched and not isinstance(
            self.interleaver.tracer, ScanningTracer
        ):
            self.dispatch()

            if isinstance(fn, torch.nn.Module):
                fn = self._module
            elif isinstance(fn, MethodType) and fn.__self__ is not None:
                # Unbind using __func__, then bind to new_instance using __get__

                new_self = (
                    self._module if isinstance(fn.__self__, torch.nn.Module) else self
                )

                fn = fn.__func__.__get__(new_self, type(new_self))

        return super().interleave(fn, *args, **kwargs)

    def __getstate__(self):
        # ``threading.Lock`` can't be pickled, and cloudpickle walks the
        # model's ``__dict__`` when serializing a ``RequestModel`` whose
        # tracer references this model. Drop the lock from the state;
        # ``__setstate__`` recreates it after unpickling.
        state = super().__getstate__()
        state.pop("_dispatch_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.dispatched = True
        # Locks can't be pickled — re-create after unpickling. The
        # post-unpickle instance is single-reader until it's handed to
        # multiple threads, so initializing here is safe.
        self._dispatch_lock = threading.Lock()
