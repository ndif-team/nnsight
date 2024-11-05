from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

from ...tracing.contexts import Tracer
from ..graph import InterventionNode, ValidatingInterventionNode, InterventionProxyType, InterventionProxy
from . import InterventionTracer, Invoker
from typing_extensions import Self
if TYPE_CHECKING:
    from .. import NNsight


class Session(Tracer[InterventionNode, InterventionProxy]):

    def __init__(self, model: "NNsight", validate: bool = False, **kwargs) -> None:

        super().__init__(
            node_class=ValidatingInterventionNode if validate else InterventionNode,
            proxy_class=model.proxy_class,
            **kwargs,
        )

        self.model = model
        
    def __enter__(self) -> Self:
        
        self.model._session = self
        
        return super().__enter__()


    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.model._session = None
        return super().__exit__(exc_type, exc_val, exc_tb)
    
    R = TypeVar('R')
    
    def apply(self, target: Callable[..., R], *args, **kwargs) -> Union[InterventionProxy, R]:
        return super().apply(target, *args, **kwargs)
    
