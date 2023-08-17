from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch.fx
from typing_extensions import override

from .fx import Tracer

if TYPE_CHECKING:
    from .Model import Model


class InvokerState:
    def __init__(self, model: "Model") -> None:
        self.model = model

        self.generation_idx: int = None
        self.batch_idx: int = None
        self.prompts: List = []
        self.tracer: Tracer = None

        self.reset()

    def reset(self):
        self.generation_idx = 0
        self.batch_idx = 0
        self.prompts.clear()
        self.tracer = Tracer(torch.fx.graph.Graph(owning_module=self.model))


class Invoker:
    """
    An Invoker represents a context window for running a single prompt which tracks
    all requested interventions applied during the invokation of the prompt

    Attributes
    ----------
        model : PreTrainedModel
        prompt : str
        args : List
        kwargs : Dict
    """

    def __init__(self, state: InvokerState, input, *args, **kwargs) -> None:
        self.state = state
        self.input = input
        self.args = args
        self.kwargs = kwargs
        self.tokens = None

    @override
    def __enter__(self) -> Invoker:
        self.state.generation_idx = 0

        inputs = self.state.model.prepare_inputs(self.input)
        self.state.model.run_graph(inputs.copy(), *self.args, **self.kwargs)

        self.tokens = [
            self.state.model.tokenizer.decode(token) for token in inputs["input_ids"][0]
        ]

        self.state.prompts.append("".join(self.tokens))

        return self

    @override
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.state.batch_idx += 1

    def next(self) -> None:
        self.state.generation_idx += 1

        self.state.model.run_graph("_", *self.args, **self.kwargs)
