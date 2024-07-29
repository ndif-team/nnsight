from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Union, Any, Dict, List, Optional

from ..tracing import protocols

if TYPE_CHECKING:
    from ..tracing.Graph import Graph
    from ..tracing.Node import Node

class ConditionalManager():
    """ A Graph attachement that manages the Conditional contexts defined within an Intervention Graph.
    
    Attributes:
        _conditional_dict (Dict): Mapping of ConditionalProtocol node name to Conditional context.
        _conditional_stack (Dict): Stack of visited Conditional contexts.
    """

    def __init__(self):
        self._conditional_dict: Dict = dict()
        self._conditional_stack: List = list()

    def push(self, conditional: Conditional) -> None:
        """ Adds the Conditional to the stack of conditional contexts.
        
        Args:
            conditional (Conditional): Conditional context.
        """

        self._conditional_dict[conditional.proxy.node.name] = conditional
        self._conditional_stack.append(conditional)

    def get(self, key: str) -> Conditional:
        """ Returns the Conditional context for a given ConditionalProtocol node name.

        Args:
            key (str): ConditionalProtocol node name.

        Returns:    
            Conditional: Conditional context.
        """

        return self._conditional_dict[key]

    def pop(self) -> None:
        """ Pops the current Conditional context from the ConditionalManager stack. """

        self._conditional_stack.pop()

    def peek(self) -> Optional[Conditional]:
        """ Gets the current Conditional context.
        
        Returns:
            Optional[Conditional]: Lastest Conditonal context if the ConditionalManager stack is non-empty.
        """

        if len(self._conditional_stack) > 0:
            return self._conditional_stack[-1]

class Conditional(AbstractContextManager):
    """ A context defined by a boolean condition, upon which the execution of all nodes defined from within is contingent. 

    Attributes:
        _graph (Graph): Intervention Graph where the Conditional context is defined.
        _condition (Conditional): Condition for the execution of the context body.
    """

    def __init__(self, graph: "Graph", condition: Union["Node", Any]):
       self.graph: "Graph" = graph
       self.condition: Union["Node", Any] = condition

    def __enter__(self) -> Conditional:

        protocols.ConditionalProtocol.add(self) 

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        protocols.ConditionalProtocol.pop_conditional(self.graph)
