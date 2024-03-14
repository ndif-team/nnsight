from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.functional_validators import AfterValidator

from ... import NNsight
from ...contexts.accum.Accumulator import Accumulator
from ...contexts.accum.Collection import Collection
from ...contexts.accum.Iterator import Iterator
from ...contexts.Tracer import Tracer
from ...tracing.Graph import Graph
from .types import *


class GraphModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["GRAPH"] = "GRAPH"

    id: int
    nodes: Dict[str, Union[NodeModel, NodeType]]

    def compile(self) -> Graph:
        graph = Graph(validate=False, graph_id=self.id)

        for node in self.nodes.values():
            node.compile(graph, self.nodes)

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

    def compile(self, model: NNsight, accumulator: Accumulator = None) -> Tracer:

        graph = self.graph.compile()
        kwargs = {key: value.compile(None, None) for key, value in self.kwargs.items()}

        batched_input = self.batched_input.compile(None, None)

        tracer = Tracer(None, model, graph=graph, **kwargs)
        tracer._batched_input = batched_input

        if accumulator is not None:

            tracer.accumulator_backend_handle(accumulator)

        return tracer


class CollectionModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["COLLECTION"] = "COLLECTION"

    collection: List[ObjectTypes]

    def compile(self, model: NNsight, accumulator: Accumulator) -> Collection:

        collection = Collection(None, accumulator=accumulator)

        collection.collection = [
            value.compile(model, accumulator) for value in self.collection
        ]

        collection.accumulator_backend_handle(accumulator)

        return collection


class IteratorModel(CollectionModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["ITERATOR"] = "ITERATOR"

    data: List[ValueTypes]

    def compile(self, model: NNsight, accumulator: Accumulator) -> Iterator:

        data = [value.compile() for value in self.data]

        iterator = Iterator(data, None, accumulator=accumulator)

        collection = super().compile(model, accumulator)

        iterator.collection = collection.collection

        return iterator


class AccumulatorModel(CollectionModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_name: Literal["ACCUMULATOR"] = "ACCUMULATOR"

    graph: Union[GraphModel, GraphType]

    def compile(self, model: NNsight) -> Accumulator:

        graph = self.graph.compile()

        accumulator = Accumulator(None, model, graph=graph)

        collection = super().compile(model, accumulator)

        accumulator.collection = collection.collection

        return accumulator


TracerType = Annotated[
    Tracer,
    AfterValidator(
        lambda value: TracerModel(
            kwargs=value._kwargs, batched_input=value._batched_input, graph=value._graph
        )
    ),
]
CollectionType = Annotated[
    Collection,
    AfterValidator(lambda value: CollectionModel(collection=value.collection)),
]
IteratorType = Annotated[
    Iterator,
    AfterValidator(
        lambda value: IteratorModel(collection=value.collection, data=value.data)
    ),
]

AccumulatorType = Annotated[
    Accumulator,
    AfterValidator(
        lambda value: AccumulatorModel(collection=value.collection, graph=value.graph)
    ),
]


ObjectTypes = Union[
    Annotated[
        Union[
            GraphModel, TracerModel, IteratorModel, AccumulatorModel, CollectionModel
        ],
        Field(discriminator="type_name"),
    ],
    Union[GraphType, TracerType, IteratorType, AccumulatorType, CollectionType],
]
