import copy
from typing import Collection

from ...tracing.graph import Graph
from ...tracing.graph import Node
from ...tracing.graph import Proxy
from . import Context
from ..protocols import VariableProtocol


class Iterator(Context):

    def __init__(self, collection: Collection, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.args = [collection]
        
    def __enter__(self) -> Proxy:
        
        super().__enter__()
        
        return VariableProtocol.add(self.graph)
        

    @classmethod
    def execute(cls, node: Node):

        graph, collection = node.args

        graph: Graph
        collection: Collection

        collection = node.prepare_inputs(collection)
        
        variable_node = next(iter(graph))
        
        for idx, value in enumerate(copy.copy(collection)):

            VariableProtocol.set(variable_node, value)

            if idx != len(collection) - 1:
                cls.defer(graph)

            graph.reset()
            graph.execute()
            
        node.set_value(None)

    @classmethod
    def defer(cls, graph: Graph):

        for node in graph:
            for dependency in node.dependencies:
                dependency.remaining_listeners += 1
