from __future__ import annotations

import weakref
from types import BuiltinFunctionType
from types import FunctionType as FuncType
from types import MethodDescriptorType
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    Strict,
    ValidationError,
    field_validator,
    model_serializer,
)
from pydantic.functional_validators import AfterValidator, BeforeValidator
from typing_extensions import Annotated, Self

from ...intervention.graph import InterventionGraph, InterventionNode
from ...tracing.graph import Graph, Node, SubGraph
from . import FUNCTIONS_WHITELIST, get_function_name

if TYPE_CHECKING:
    from ... import NNsight

FUNCTION = Union[BuiltinFunctionType, FuncType, MethodDescriptorType, type]
PRIMITIVE = Union[int, float, str, bool, None]


class DeserializeHandler:

    def __init__(
        self,
        memo,
        model: "NNsight"
    ) -> None:

        self.memo = memo
        self.model = model
        self.graph = Graph(node_class=InterventionNode)



MEMO = {}


class BaseNNsightModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["TYPE_NAME"]

    @classmethod
    def to_model(cls, value: Any) -> Self:
        raise NotImplementedError()

    def deserialize(self, handler: DeserializeHandler):
        raise NotImplementedError()


def try_deserialize(value: Union[BaseNNsightModel, Any], handler: DeserializeHandler):

    if isinstance(value, BaseNNsightModel):

        return value.deserialize(handler)

    return value


def memoized(fn):

    def inner(value):

        model = fn(value)

        _id = id(value)

        MEMO[_id] = model

        return MemoReferenceModel(id=_id)

    return inner


### Custom Pydantic types for all supported base types
class NodeModel(BaseNNsightModel):

    type_name: Literal["NODE"] = "NODE"

    target: ValueTypes
    args: List[ValueTypes] = []
    kwargs: Dict[str, ValueTypes] = {}

    @staticmethod
    @memoized
    def to_model(value: Node) -> Self:

        return NodeModel(target=value.target, args=value.args, kwargs=value.kwargs)

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):

        dump = handler(self)

        if not self.kwargs:

            dump.pop("kwargs")

        if not self.args:

            dump.pop("args")

        return dump

    def deserialize(self, handler: DeserializeHandler) -> Node:

        return handler.graph.create(
            self.target.deserialize(handler),
            *[try_deserialize(value, handler) for value in self.args],
            **{
                key: try_deserialize(value, handler)
                for key, value in self.kwargs.items()
            }
        ).node

class TensorModel(BaseNNsightModel):

    type_name: Literal["TENSOR"] = "TENSOR"

    values: List
    dtype: str

    @staticmethod
    @memoized
    def to_model(value: torch.Tensor) -> Self:

        return TensorModel(values=value.tolist(), dtype=str(value.dtype).split(".")[-1])

    def deserialize(self, handler: DeserializeHandler) -> torch.Tensor:
        dtype = getattr(torch, self.dtype)
        return torch.tensor(self.values, dtype=dtype)


class SliceModel(BaseNNsightModel):

    type_name: Literal["SLICE"] = "SLICE"

    start: ValueTypes
    stop: ValueTypes
    step: ValueTypes

    @staticmethod
    @memoized
    def to_model(value: slice) -> Self:

        return SliceModel(start=value.start, stop=value.stop, step=value.step)

    def deserialize(self, handler: DeserializeHandler) -> slice:

        return slice(
            try_deserialize(self.start, handler),
            try_deserialize(self.stop, handler),
            try_deserialize(self.step, handler),
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

    @staticmethod
    def to_model(value: List) -> Self:

        return ListModel(values=value)

    def deserialize(self, handler: DeserializeHandler) -> list:
        return [try_deserialize(value, handler) for value in self.values]


class TupleModel(BaseNNsightModel):

    type_name: Literal["TUPLE"] = "TUPLE"

    values: List[ValueTypes]

    @staticmethod
    def to_model(value: Tuple) -> Self:

        return TupleModel(values=value)

    def deserialize(self, handler: DeserializeHandler) -> tuple:
        return tuple([try_deserialize(value, handler) for value in self.values])


class DictModel(BaseNNsightModel):

    type_name: Literal["DICT"] = "DICT"

    values: Dict[str, ValueTypes]

    @staticmethod
    def to_model(value: Dict) -> Self:

        return DictModel(values=value)

    def deserialize(self, handler: DeserializeHandler) -> dict:
        return {
            key: try_deserialize(value, handler) for key, value in self.values.items()
        }


class FunctionWhitelistError(Exception):
    pass


class FunctionModel(BaseNNsightModel):

    type_name: Literal["FUNCTION"] = "FUNCTION"

    function_name: str
    
    @staticmethod
    def to_model(value:FUNCTION):
        
        model = FunctionModel(function_name=get_function_name(value))
        
        FunctionModel.check_function_whitelist(model.function_name)
        
        return model

    @classmethod
    def check_function_whitelist(cls, qualname: str) -> str:
        if qualname not in FUNCTIONS_WHITELIST:
            raise FunctionWhitelistError(
                f"Function with name `{qualname}` not in function whitelist."
            )

        return qualname

    def deserialize(self, handler: DeserializeHandler) -> FUNCTION:

        FunctionModel.check_function_whitelist(self.function_name)

        return FUNCTIONS_WHITELIST[self.function_name]


class GraphModel(BaseNNsightModel):

    type_name: Literal["GRAPH"] = "GRAPH"

    # We have a reference to the real Graph in the pydantic to be used by optimization logic
    graph: Graph = Field(exclude=True, default=None, validate_default=False)

    nodes: List[Union[MemoReferenceModel, NodeType]]

    @staticmethod
    def to_model(value: Graph) -> Self:

        return GraphModel(graph=value, nodes=value.nodes)

    def deserialize(self, handler: DeserializeHandler) -> Graph:

        for node in self.nodes:

            node.deserialize(handler)

        return handler.graph


class SubGraphModel(BaseNNsightModel):

    type_name: Literal["SUBGRAPH"] = "SUBGRAPH"

    subset: List[int]

    @staticmethod
    def to_model(value: SubGraph) -> Self:

        return SubGraphModel(subset=value.subset)

    def deserialize(self, handler: DeserializeHandler) -> Graph:

        value = SubGraph(handler.graph, subset=self.subset)
        
        for node in value:
            node.graph = value
            
        return value


class InterventionGraphModel(SubGraphModel):

    type_name: Literal["INTERVENTIONGRAPH"] = "INTERVENTIONGRAPH"

    @staticmethod
    def to_model(value: InterventionGraph) -> Self:

        return InterventionGraphModel(subset=value.subset)

    def deserialize(self, handler: DeserializeHandler) -> Graph:
        value = InterventionGraph(handler.graph, model=handler.model, subset=self.subset)
    
        for node in value:
            node.graph = value
            
        return value


class MemoReferenceModel(BaseNNsightModel):

    type_name: Literal["REFERENCE"] = "REFERENCE"

    id: int

    def deserialize(self, handler: DeserializeHandler):
        
        value = try_deserialize(handler.memo[self.id], handler)

        handler.memo[self.id] = value

        return value


### Define Annotated types to convert objects to their custom Pydantic counterpart

GraphType = Annotated[
    Graph,
    AfterValidator(GraphModel.to_model),
]

SubGraphType = Annotated[
    SubGraph,
    AfterValidator(SubGraphModel.to_model),
]

InterventionGraphType = Annotated[
    InterventionGraph,
    AfterValidator(InterventionGraphModel.to_model),
]

TensorType = Annotated[torch.Tensor, AfterValidator(TensorModel.to_model)]

SliceType = Annotated[
    slice,
    AfterValidator(SliceModel.to_model),
]

EllipsisType = Annotated[
    type(...),  # It will be better to use EllipsisType, but it requires python>=3.10
    AfterValidator(lambda _: EllipsisModel()),
]


ListType = Annotated[list, AfterValidator(ListModel.to_model)]

TupleType = Annotated[
    tuple,
    Strict(),
    AfterValidator(TupleModel.to_model),
]

DictType = Annotated[dict, AfterValidator(DictModel.to_model)]

FunctionType = Annotated[
    FUNCTION,
    AfterValidator(FunctionModel.to_model),
]

NodeType = Annotated[
    Node,
    AfterValidator(NodeModel.to_model),
]


def check_memo(object: Any):

    _id = id(object)

    if _id in MEMO:

        return MemoReferenceModel(id=_id)

    raise ValueError()


MemoType = Annotated[object, BeforeValidator(check_memo)]

### Register all custom Pydantic objects to convert objects to
TOTYPES = Annotated[
    Union[
        MemoReferenceModel,
        NodeModel,
        SliceModel,
        TensorModel,
        TupleModel,
        ListModel,
        DictModel,
        FunctionModel,
        EllipsisModel,
        InterventionGraphModel,
        SubGraphModel,
        GraphModel,
    ],
    Field(discriminator="type_name"),
]
### Register all Annotated types objects to convert objects from
FROMTYPES = Annotated[
    Union[
        MemoType,
        NodeType,
        InterventionGraphType,
        SubGraphType,
        GraphType,
        FunctionType,
        SliceType,
        TensorType,
        TupleType,
        ListType,
        DictType,
        EllipsisType,
    ],
    Field(union_mode="left_to_right"),
]

### Final registration
ValueTypes = Union[
    PRIMITIVE,
    TOTYPES,
    FROMTYPES,
]
