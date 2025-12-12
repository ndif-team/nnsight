from typing import TYPE_CHECKING, Any

from ..tracing.globals import Globals
from ..tracing.util import wrap_exception
from .base import Backend

if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any


class ExecutionBackend(Backend):

    def __call__(self, tracer: Tracer):

        fn = super().__call__(tracer)

        try:
            Globals.enter()
            return tracer.execute(fn)
        except Exception as e:

            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()
