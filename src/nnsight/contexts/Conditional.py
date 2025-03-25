from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Any, Union

from ..tracing import protocols

if TYPE_CHECKING:
    from ..tracing.Node import Node
    from ..tracing.Graph import Graph
    from ..intervention import InterventionProxy

class ConditionalManager():
    """ A Graph attachement that manages the Conditional contexts defined within an Intervention Graph.
    
    Attributes:
        _conditional_dict (Dict[str, Node]): Mapping of ConditionalProtocol node name to Conditional context.
        _conditioned_nodes_dict (Dict[str, Set[Node]]): Mapping of ConditionalProtocol node name to all the Nodes conditiones by it.
        _conditional_stack (Dict): Stack of visited Conditional contexts' ConditonalProtocol nodes.
    """

    def __init__(self):
        self._conditional_nodes_dict: Dict[str, Node] = dict()
        self._conditioned_nodes_dict: Dict[str, Set[Node]] = dict()
        self._conditional_nodes_stack: List[Node] = list()

    def push(self, conditional_node: "Node") -> None:
        """ Adds the Conditional to the stack of Conditional contexts.
        
        Args:
            conditional_node (Node): ConditionalProtocol node.
        """

        self._conditional_nodes_dict[conditional_node.name] = conditional_node
        self._conditioned_nodes_dict[conditional_node.name] = set()
        self._conditional_nodes_stack.append(conditional_node)

    def get(self, key: str) -> Conditional:
        """ Returns a ConditionalProtocol node.

        Args:
            key (str): ConditionalProtocol node name.

        Returns:    
            Node: ConditionalProtocol node.
        """

        return self._conditional_nodes_dict[key]

    def pop(self) -> None:
        """ Pops the ConditionalProtocol node of the current Conditional context from the ConditionalManager stack. """

        self._conditional_nodes_stack.pop()

    def peek(self) -> Optional["Node"]:
        """ Gets the current Conditional context's ConditionalProtocol node.
        
        Returns:
            Optional[Node]: Lastest ConditonalProtocol node if the ConditionalManager stack is non-empty.
        """

        if len(self._conditional_nodes_stack) > 0:
            return self._conditional_nodes_stack[-1]
        
    def add_conditioned_node(self, node: "Node") -> None:
        """  Adding a Node to the set of conditioned nodes by the current Conditonal context.

        Args:
            - node (Node): A node conditioned by the latest Conditional context.
        """

        self._conditioned_nodes_dict[self.peek().name].add(node)

    def is_node_conditioned(self, node: "Node") -> bool:
        """ Returns True if the Node argument is conditioned by the current Conditional context.
        
        Args:
            - node (Node): Node.

        Returns:
            bool: Whether the Node is conditioned.
        """

        curr_conditioned_nodes_set = self._conditioned_nodes_dict[self.peek().name]

        return (node in curr_conditioned_nodes_set)
    

class Conditional(AbstractContextManager):
    """ A context defined by a boolean condition, upon which the execution of all nodes defined from within is contingent. 

    Attributes:
        _graph (Graph): Conditional Context graph.
        _condition (Union[InterventionProxy, Any]): Condition.
    """

    def __init__(self, graph: "Graph", condition: Union["InterventionProxy", Any]):
       self._graph = graph
       self._condition: Union["InterventionProxy", Any] = condition

    def __enter__(self) -> Conditional:

        conditional_node = protocols.ConditionalProtocol.add(self._graph, self._condition).node

        protocols.ConditionalProtocol.push_conditional(conditional_node)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        protocols.ConditionalProtocol.pop_conditional(self._graph)
