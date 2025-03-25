from __future__ import annotations

import weakref
from types import BuiltinFunctionType
from types import FunctionType as FuncType
from types import MethodDescriptorType
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import (BaseModel, ConfigDict, Field, Strict, field_validator,
                      model_serializer)
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from ...contexts.session.Iterator import Iterator
from ...contexts.session.Session import Session
from ...contexts.Tracer import Tracer
from ...models.NNsightModel import NNsight
from ...tracing import protocols
from ...tracing.Bridge import Bridge
from ...tracing.Graph import Graph
from ...tracing.Node import Node
from . import FUNCTIONS_WHITELIST, get_function_name


class DeserializeHandler:

    def __init__(
        self,
        graph: Graph = None,
        nodes: Dict[str, Union[NodeModel, NodeType]] = None,
        model: NNsight = None,
        bridge: Bridge = None,
    ) -> None:

        self.graph = graph
        self.nodes = nodes
        self.model = model
        self.bridge = bridge


FUNCTION = Union[BuiltinFunctionType, FuncType, MethodDescriptorType, type]
PRIMITIVE = Union[int, float, str, bool, None]


class BaseNNsightModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["TYPE_NAME"]

    def deserialize(self, handler: DeserializeHandler):
        raise NotImplementedError()

def try_deserialize(value: BaseNNsightModel | Any, handler: DeserializeHandler):
    
    if isinstance(value, BaseNNsightModel):
        
        return value.deserialize(handler)
    
    return value


### Custom Pydantic types for all supported base types
class NodeModel(BaseNNsightModel):

    type_name: Literal["NODE"] = "NODE"

    class Reference(BaseNNsightModel):
        type_name: Literal["NODE_REFERENCE"] = "NODE_REFERENCE"

        name: str

        def deserialize(self, handler: DeserializeHandler) -> Node:
            return handler.nodes[self.name].deserialize(handler)

    name: str
    target: Union[FunctionModel, FunctionType]
    args: List[ValueTypes] = []
    kwargs: Dict[str, ValueTypes] = {}
    condition: Union[
        NodeReferenceType, NodeModel.Reference, None
    ] = None
    
    @model_serializer(mode='wrap')
    def serialize_model(self, handler):
            
        dump = handler(self)
        
        if self.condition is None:
            
            dump.pop('condition')
            
        if not self.kwargs:
            
            dump.pop('kwargs')
            
        if not self.args:
            
            dump.pop('args')
            
        return dump

    def deserialize(self, handler: DeserializeHandler) -> Node:

        if self.name in handler.graph.nodes:
            return handler.graph.nodes[self.name]

        node = handler.graph.create(
            proxy_value=None,
            target=self.target.deserialize(handler),
            args=[try_deserialize(value, handler) for value in self.args],
            kwargs={
                key: try_deserialize(value, handler) for key, value in self.kwargs.items()
            },
            name=self.name,
        ).node

        node.cond_dependency = try_deserialize(self.condition, handler)
        
        if isinstance(node.cond_dependency, Node):
            node.cond_dependency.listeners.append(weakref.proxy(node))

        if isinstance(node.target, type) and issubclass(
            node.target, protocols.Protocol
        ):

            node.target.compile(node)

        return node

class TensorModel(BaseNNsightModel):

    type_name: Literal["TENSOR"] = "TENSOR"

    values: List
    dtype: str

    def deserialize(self, handler: DeserializeHandler) -> torch.Tensor:
        dtype = getattr(torch, self.dtype)
        return torch.tensor(self.values, dtype=dtype)


class SliceModel(BaseNNsightModel):

    type_name: Literal["SLICE"] = "SLICE"

    start: ValueTypes
    stop: ValueTypes
    step: ValueTypes

    def deserialize(self, handler: DeserializeHandler) -> slice:

        return slice(
            try_deserialize(self.start, handler),
            try_deserialize(self.stop, handler),
            try_deserialize(self.step, handler)
        )


class EllipsisModel(BaseNNsightModel):

    type_name: Literal["ELLIPSIS"] = "ELLIPSIS"

    def deserialize(
        self, handler: DeserializeHandler
    ) -> type(
        ...
    ):  # It will be better to use EllipsisType, but it requires python>=3.10
        return ...


class ListModel(BaseNNsightModel):

    type_name: Literal["LIST"] = "LIST"

    values: List[ValueTypes]

    def deserialize(self, handler: DeserializeHandler) -> list:
        return [try_deserialize(value, handler) for value in self.values]


class TupleModel(BaseNNsightModel):

    type_name: Literal["TUPLE"] = "TUPLE"

    values: List[ValueTypes]

    def deserialize(self, handler: DeserializeHandler) -> tuple:
        return tuple([try_deserialize(value, handler) for value in self.values])


class DictModel(BaseNNsightModel):

    type_name: Literal["DICT"] = "DICT"

    values: Dict[str, ValueTypes]

    def deserialize(self, handler: DeserializeHandler) -> dict:
        return {key: try_deserialize(value, handler) for key, value in self.values.items()}


class FunctionWhitelistError(Exception):
    pass


class FunctionModel(BaseNNsightModel):

    type_name: Literal["FUNCTION"] = "FUNCTION"

    function_name: str

    @field_validator("function_name")
    @classmethod
    def check_function_whitelist(cls, qualname: str) -> str:
        if qualname not in FUNCTIONS_WHITELIST:
            raise FunctionWhitelistError(
                f"Function with name `{qualname}` not in function whitelist."
            )

        return qualname

    def deserialize(self, handler: DeserializeHandler) -> FUNCTION:
        return FUNCTIONS_WHITELIST[self.function_name]


class GraphModel(BaseNNsightModel):

    type_name: Literal["GRAPH"] = "GRAPH"

    id: int
    sequential: bool
    nodes: Dict[str, Union["NodeModel", "NodeType"]]

    def deserialize(self, handler: DeserializeHandler) -> Graph:

        graph = Graph(validate=False, sequential=self.sequential, graph_id=self.id)

        handler.graph = graph
        handler.nodes = self.nodes

        # To preserve order
        nodes = {}

        for node_name, node in self.nodes.items():

            node.deserialize(handler)

            # To preserve order
            nodes[node_name] = graph.nodes[node_name]

        # To preserve order
        graph.nodes = nodes

        return graph


class TracerModel(BaseNNsightModel):

    type_name: Literal["TRACER"] = "TRACER"

    kwargs: Dict[str, ValueTypes]
    invoker_inputs: List[ValueTypes]
    graph: Union[GraphModel, GraphType]

    def deserialize(self, handler: DeserializeHandler) -> Tracer:

        _graph = handler.graph
        _nodes = handler.nodes

        graph = self.graph.deserialize(handler)

        handler.graph = graph

        kwargs = {key: try_deserialize(value, handler) for key, value in self.kwargs.items()}

        invoker_inputs = [
            try_deserialize(invoker_input, handler) for invoker_input in self.invoker_inputs
        ]

        tracer = Tracer(
            None, handler.model, bridge=handler.bridge, graph=graph, **kwargs
        )
        tracer._invoker_inputs = invoker_inputs

        handler.graph = _graph
        handler.nodes = _nodes

        return tracer


class IteratorModel(BaseNNsightModel):

    type_name: Literal["ITERATOR"] = "ITERATOR"

    data: ValueTypes

    graph: Union[GraphModel, GraphType]

    def deserialize(self, handler: DeserializeHandler) -> Iterator:

        _graph = handler.graph
        _nodes = handler.nodes

        graph = self.graph.deserialize(handler)

        handler.graph = graph

        data = try_deserialize(self.data, handler)

        iterator = Iterator(data, None, bridge=handler.bridge, graph=graph)

        handler.graph = _graph
        handler.nodes = _nodes

        return iterator


class SessionModel(BaseNNsightModel):

    type_name: Literal["SESSION"] = "SESSION"

    graph: Union[GraphModel, GraphType]

    def deserialize(self, handler: DeserializeHandler) -> Session:

        bridge = Bridge()

        handler.bridge = bridge

        graph = self.graph.deserialize(handler)

        bridge.add(graph)

        session = Session(None, handler.model, bridge=bridge, graph=graph)

        return session


### Define Annotated types to convert objects to their custom Pydantic counterpart

GraphType = Annotated[
    Graph,
    AfterValidator(
        lambda value: GraphModel(
            id=value.id, sequential=value.sequential, nodes=value.nodes
        )
    ),
]

TensorType = Annotated[
    torch.Tensor,
    AfterValidator(
        lambda value: TensorModel(
            values=value.tolist(), dtype=str(value.dtype).split(".")[-1]
        )
    ),
]

SliceType = Annotated[
    slice,
    AfterValidator(
        lambda value: SliceModel(start=value.start, stop=value.stop, step=value.step)
    ),
]

EllipsisType = Annotated[
    type(...),  # It will be better to use EllipsisType, but it requires python>=3.10
    AfterValidator(lambda value: EllipsisModel()),
]


ListType = Annotated[list, AfterValidator(lambda value: ListModel(values=value))]

TupleType = Annotated[
    tuple, Strict(), AfterValidator(lambda value: TupleModel(values=list(value)))
]

DictType = Annotated[dict, AfterValidator(lambda value: DictModel(values=value))]

FunctionType = Annotated[
    FUNCTION,
    AfterValidator(lambda value: FunctionModel(function_name=get_function_name(value))),
]

NodeReferenceType = Annotated[
    Node, AfterValidator(lambda value: NodeModel.Reference(name=value.name))
]

NodeType = Annotated[
    Node,
    AfterValidator(
        lambda value: NodeModel(
            name=value.name,
            target=value.target,
            args=value.args,
            kwargs=value.kwargs,
            condition=value.cond_dependency,
        )
    ),
]

TracerType = Annotated[
    Tracer,
    AfterValidator(
        lambda value: TracerModel(
            kwargs=value._kwargs,
            invoker_inputs=value._invoker_inputs,
            graph=value.graph,
        )
    ),
]

IteratorType = Annotated[
    Iterator,
    AfterValidator(lambda value: IteratorModel(graph=value.graph, data=value.data)),
]

SessionType = Annotated[
    Session,
    AfterValidator(lambda value: SessionModel(graph=value.graph)),
]

### Register all custom Pydantic objects to convert objects to
TOTYPES = Union[
    TracerModel,
    IteratorModel,
    SessionModel,
    NodeModel.Reference,
    SliceModel,
    TensorModel,
    TupleModel,
    ListModel,
    DictModel,
    EllipsisModel,
]
### Register all Annotated types objects to convert objects from
FROMTYPES = Union[
    TracerType,
    IteratorType,
    SessionType,
    NodeReferenceType,
    SliceType,
    TensorType,
    TupleType,
    ListType,
    DictType,
    EllipsisType,
]

### Final registration
ValueTypes = Union[
    PRIMITIVE,
    Annotated[
        TOTYPES,
        Field(discriminator="type_name"),
    ],
    FROMTYPES,
]
