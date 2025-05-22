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

        tracer.compile()

        source = "".join(tracer.info.source)

        code_obj = compile(source, tracer.info.filename, "exec")

        local_namespace = {}

        # Execute the function definition in the local namespace
        exec(
            code_obj,
            {**tracer.info.frame.f_globals, **tracer.info.frame.f_locals},
            local_namespace,
        )

        fn = list(local_namespace.values())[-1]

        # TODO maybe move it tracer __exit__
        try:
            Globals.enter()
            tracer.execute(fn)
        except Exception as e:

            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()
