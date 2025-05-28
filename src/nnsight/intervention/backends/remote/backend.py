import inspect
from typing import TYPE_CHECKING, Any, Callable

import dill

from .. import ExecutionBackend
from ..base import Backend
from .schema import Request
from .utils import (
    WHITELISTED_MODULES_DESERIALIZATION,
    Protector,
    WHITELISTED_MODULES,
    ProtectorEscape,
)
from .sandbox import run
from ...tracing.globals import Globals

if TYPE_CHECKING:
    from ...tracing.tracer import Tracer
else:
    Tracer = Any


class RemoteBackend(Backend):

    def __call__(self, tracer: Tracer):

        fn = super().__call__(tracer)

        request = Request(
            model_key=tracer.model.to_model_key(), intervention=fn, tracer=tracer
        )

        dill.settings["recurse"] = True

        with open("request.pkl", "wb") as f:
            dill.dump(request, f)


class RemoteExecutionBackend(Backend):

    def __init__(self, model: Any, fn: Callable):
        self.model = model
        self.fn = fn

    def __call__(self, tracer: Tracer):

        tracer.__setmodel__(self.model)

        protector = Protector(WHITELISTED_MODULES)
        escape = ProtectorEscape(protector)

        Globals.enter()

        with protector:
            with escape:
                run(tracer, self.fn)

        Globals.exit()

        return {
            key: value
            for key, value in tracer.info.frame.f_locals.items()
            if not key
            in {
                "__nnsight_tracer__",
                "__nnsight_model__",
                "tracer",
                "fn",
                "__nnsight_tracing_info__",
            }
        }
