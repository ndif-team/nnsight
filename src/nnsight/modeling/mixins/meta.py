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

        Calls :meth:`_load` with the saved constructor arguments and
        updates the Envoy tree to reflect the newly loaded module.
        After this call, :attr:`dispatched` is ``True``.
        """

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
            self._interleaver.tracer, ScanningTracer
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

    def __setstate__(self, state):
        super().__setstate__(state)
        self.dispatched = True
