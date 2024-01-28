from __future__ import annotations

from types import BuiltinFunctionType
from types import FunctionType as FuncType
from types import MethodDescriptorType
from typing import Dict, List, Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from ...tracing.Graph import Graph
from ...tracing.Node import Node
from . import FUNCTIONS_WHITELIST, get_function_name

FUNCTION = Union[BuiltinFunctionType, FuncType, MethodDescriptorType, str]
PRIMITIVE = Union[int, float, str, bool, None]


class NodeModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class Reference(BaseModel):
        type_name: Literal["NODE_REFERENCE"] = "NODE_REFERENCE"

        name: str

        def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> Node:
            return nodes[self.name].compile(graph, nodes)

    name: str
    target: Union[FunctionModel, FunctionType]
    args: List[ValueTypes]
    kwargs: Dict[str, ValueTypes]

    def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> Node:
        if self.name in graph.nodes:
            return graph.nodes[self.name]

        return graph.add(
            value=None,
            target=self.target.compile(graph, nodes),
            args=[value.compile(graph, nodes) for value in self.args],
            kwargs={
                key: value.compile(graph, nodes) for key, value in self.kwargs.items()
            },
            name=self.name,
        )


class PrimitiveModel(BaseModel):
    type_name: Literal["PRIMITIVE"] = "PRIMITIVE"
    value: PRIMITIVE

    def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> PRIMITIVE:
        return self.value


class TensorModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["TENSOR"] = "TENSOR"

    values: List
    dtype: str

    def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> torch.Tensor:
        dtype = getattr(torch, self.dtype)
        return torch.tensor(self.values, dtype=dtype)


class SliceModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["SLICE"] = "SLICE"

    start: ValueTypes
    stop: ValueTypes
    step: ValueTypes

    def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> slice:
        return slice(
            self.start.compile(graph, nodes),
            self.stop.compile(graph, nodes),
            self.step.compile(graph, nodes),
        )


class ListModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["LIST"] = "LIST"

    values: List[ValueTypes]

    def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> list:
        return [value.compile(graph, nodes) for value in self.values]


class TupleModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["TUPLE"] = "TUPLE"

    values: List[ValueTypes]

    def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> tuple:
        return tuple([value.compile(graph, nodes) for value in self.values])


class DictModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["DICT"] = "DICT"

    values: Dict[str, ValueTypes]

    def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> dict:
        return {key: value.compile(graph, nodes) for key, value in self.values.items()}


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

    def compile(self, graph: Graph, nodes: Dict[str, NodeModel]) -> FUNCTION:
        return FUNCTIONS_WHITELIST[self.function_name]


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

ListType = Annotated[list, AfterValidator(lambda value: ListModel(values=value))]

TupleType = Annotated[
    tuple, AfterValidator(lambda value: TupleModel(values=list(value)))
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
            name=value.name, target=value.target, args=value.args, kwargs=value.kwargs
        )
    ),
]

ValueTypes = Union[
    Annotated[
        Union[
            NodeModel.Reference,
            SliceModel,
            TensorModel,
            PrimitiveModel,
            ListModel,
            TupleModel,
            DictModel,
        ],
        Field(discriminator="type_name"),
    ],
    Union[
        NodeReferenceType,
        SliceType,
        TensorType,
        PrimitiveType,
        ListType,
        TupleType,
        DictType,
    ],
]
