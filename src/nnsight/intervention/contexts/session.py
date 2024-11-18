from typing import TYPE_CHECKING, Optional

from typing_extensions import Self

from ..graph import (InterventionNode, InterventionProxy,
                     ValidatingInterventionNode)
from . import InterventionTracer

if TYPE_CHECKING:
    from .. import NNsight


class Session(InterventionTracer[InterventionNode, InterventionProxy]):
    """A Session simply allows grouping multiple Tracers in one computation graph.
    """

    def __init__(self, model: "NNsight", validate: bool = False, debug:Optional[bool] = None, **kwargs) -> None:

        super().__init__(
            node_class=ValidatingInterventionNode if validate else InterventionNode,
            proxy_class=model.proxy_class,
            debug=debug,
            **kwargs,
        )

        self.model = model
        
    def __enter__(self) -> Self:
        
        self.model._session = self
        
        return super().__enter__()


    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.model._session = None
        return super().__exit__(exc_type, exc_val, exc_tb)
