from __future__ import annotations

import weakref
from types import BuiltinFunctionType
from types import FunctionType as FuncType
from types import MethodDescriptorType
from typing import Dict, List, Literal, Union, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field, Strict, field_validator
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


class NodeModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class Reference(BaseModel):
        type_name: Literal["NODE_REFERENCE"] = "NODE_REFERENCE"

        name: str

        def deserialize(self, handler: DeserializeHandler) -> Node:
            return handler.nodes[self.name].deserialize(handler)

    name: str
    target: Union[FunctionModel, FunctionType]
    args: List[ValueTypes]
    kwargs: Dict[str, ValueTypes]
    condition: Union[NodeReferenceType, NodeModel.Reference, PrimitiveModel, PrimitiveType]

    def deserialize(self, handler: DeserializeHandler) -> Node:

        if self.name in handler.graph.nodes:
            return handler.graph.nodes[self.name]

        node = handler.graph.create(
            proxy_value=None,
            target=self.target.deserialize(handler),
            args=[value.deserialize(handler) for value in self.args],
            kwargs={
                key: value.deserialize(handler) for key, value in self.kwargs.items()
            },
            name=self.name,
        ).node

        node.cond_dependency = self.condition.deserialize(handler)
        if isinstance(node.cond_dependency, Node):
            node.cond_dependency.listeners.append(weakref.proxy(node))

        if isinstance(node.target, type) and issubclass(
            node.target, protocols.Protocol
        ):

            node.target.compile(node)

        return node


class PrimitiveModel(BaseModel):
    type_name: Literal["PRIMITIVE"] = "PRIMITIVE"
    value: PRIMITIVE

    def deserialize(self, handler: DeserializeHandler) -> PRIMITIVE:
        return self.value


class TensorModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["TENSOR"] = "TENSOR"

    values: List
    dtype: str

    def deserialize(self, handler: DeserializeHandler) -> torch.Tensor:
        dtype = getattr(torch, self.dtype)
        return torch.tensor(self.values, dtype=dtype)


class SliceModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["SLICE"] = "SLICE"

    start: ValueTypes
    stop: ValueTypes
    step: ValueTypes

    def deserialize(self, handler: DeserializeHandler) -> slice:

        return slice(
            self.start.deserialize(handler),
            self.stop.deserialize(handler),
            self.step.deserialize(handler),
        )


class EllipsisModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type_name: Literal["ELLIPSIS"] = "ELLIPSIS"

    def deserialize(
        self, handler: DeserializeHandler
    ) -> type(
        ...
    ):  # It will be better to use EllipsisType, but it requires python>=3.10
        return ...


class ListModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["LIST"] = "LIST"

    values: List[ValueTypes]

    def deserialize(self, handler: DeserializeHandler) -> list:
        return [value.deserialize(handler) for value in self.values]


class TupleModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["TUPLE"] = "TUPLE"

    values: List[ValueTypes]

    def deserialize(self, handler: DeserializeHandler) -> tuple:
        return tuple([value.deserialize(handler) for value in self.values])


class DictModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["DICT"] = "DICT"

    values: Dict[str, ValueTypes]

    def deserialize(self, handler: DeserializeHandler) -> dict:
        return {key: value.deserialize(handler) for key, value in self.values.items()}


class FunctionWhitelistError(Exception):
    pass


class FunctionModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class GraphModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["GRAPH"] = "GRAPH"

    id: int
    nodes: Dict[str, Union["NodeModel", "NodeType"]]

    def deserialize(self, handler: DeserializeHandler) -> Graph:

        graph = Graph(validate=False, graph_id=self.id)

        handler.graph = graph
        handler.nodes = self.nodes

        for node in self.nodes.values():
            node.deserialize(handler)

        return graph


GraphType = Annotated[
    Graph, AfterValidator(lambda value: GraphModel(id=value.id, nodes=value.nodes))
]


class TracerModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["TRACER"] = "TRACER"

    kwargs: Dict[str, ValueTypes]
    batched_input: ValueTypes
    graph: Union[GraphModel, GraphType]

    def deserialize(self, handler: DeserializeHandler) -> Tracer:

        kwargs = {key: value.deserialize(handler) for key, value in self.kwargs.items()}

        batched_input = self.batched_input.deserialize(handler)

        graph = self.graph.deserialize(handler)

        tracer = Tracer(
            None, handler.model, bridge=handler.bridge, graph=graph, **kwargs
        )
        tracer._batched_input = batched_input

        return tracer


class IteratorModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["ITERATOR"] = "ITERATOR"

    data: List[ValueTypes]

    graph: Union[GraphModel, GraphType]

    def deserialize(self, handler: DeserializeHandler) -> Iterator:

        data = [value.deserialize(handler) for value in self.data]

        graph = self.graph.deserialize(handler)

        iterator = Iterator(data, None, bridge=handler.bridge, graph=graph)

        return iterator


class SessionModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["SESSION"] = "SESSION"

    graph: Union[GraphModel, GraphType]

    def deserialize(self, handler: DeserializeHandler) -> Session:

        bridge = Bridge()

        handler.bridge = bridge

        graph = self.graph.deserialize(handler)

        bridge.add(graph)

        session = Session(None, handler.model, bridge=bridge, graph=graph)

        return session


PrimitiveType = Annotated[
    PRIMITIVE, AfterValidator(lambda value: PrimitiveModel(value=value))
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
            name=value.name, target=value.target, args=value.args, kwargs=value.kwargs, condition=value.cond_dependency
        )
    ),
]

TracerType = Annotated[
    Tracer,
    AfterValidator(
        lambda value: TracerModel(
            kwargs=value._kwargs, batched_input=value._batched_input, graph=value.graph
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

TOTYPES = Union[
    TracerModel,
    IteratorModel,
    SessionModel,
    NodeModel.Reference,
    SliceModel,
    TensorModel,
    PrimitiveModel,
    TupleModel,
    ListModel,
    DictModel,
    EllipsisModel,
]
FROMTYPES = Union[
    TracerType,
    IteratorType,
    SessionType,
    NodeReferenceType,
    SliceType,
    TensorType,
    PrimitiveType,
    TupleType,
    ListType,
    DictType,
    EllipsisType,
]

# Register all values
ValueTypes = Union[
    Annotated[
        TOTYPES,
        Field(discriminator="type_name"),
    ],
    FROMTYPES,
]
