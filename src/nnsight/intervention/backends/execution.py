from typing import TYPE_CHECKING, Any

from ..tracing.util import wrap_exception
from .base import Backend


if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any

from ..tracing.globals import _ensure_mounted


class ExecutionBackend(Backend):

    def __call__(self, tracer: Tracer):

        fn = super().__call__(tracer)

        try:
            # Eager one-time mount: `.save()` is dispatched via this C-level mount so any
            # object can carry the method, and the user calls `.save()` directly inside
            # trace bodies — meaning the mount must already be in place by the time user
            # code runs. No tracer-entry hook is available before the user types
            # `attn.output.save()`, so we mount at import.
            _ensure_mounted()

            return tracer.execute(fn)
        except Exception as e:

            raise wrap_exception(e, tracer.info) from None
