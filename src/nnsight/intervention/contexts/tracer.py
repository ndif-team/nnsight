import inspect
import weakref
from functools import wraps
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    TypeVar, Union)

from ...tracing.contexts import Tracer
from ...tracing.graph import Proxy
from ..graph import (InterventionNodeType, InterventionProxy,
                     InterventionProxyType)
from . import LocalContext


class InterventionTracer(Tracer[InterventionNodeType, InterventionProxyType]):

    R = TypeVar("R")

    def apply(
        self, target: Callable[..., R], *args, **kwargs
    ) -> Union[InterventionProxy, R]:
        return super().apply(target, *args, **kwargs)

    def local(self, fn: Optional[Callable] = None):

        if fn is None:

            return LocalContext(parent=self.graph)

        elif inspect.isroutine(fn):

            @wraps(fn)
            def inner(*args, **kwargs):

                with LocalContext(parent=self.graph) as context:

                    return context.apply(fn, *args, **kwargs)

        else:

            # TODO: error
            pass
