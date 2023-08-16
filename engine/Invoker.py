from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
    from .Model import Model


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

    def __init__(self, model: "Model", prompt: str, *args, **kwargs) -> None:
        self.model = model
        self.prompt = prompt
        self.args = args
        self.kwargs = kwargs

    @override
    def __enter__(self) -> Invoker:
        self.model.idx_tracker.generation_idx = 0

        self.model.prompts.append(self.prompt)

        inputs = self.model.run_graph(self.prompt, *self.args, **self.kwargs)

        self.tokens = [
            self.model.tokenizer.decode(token) for token in inputs["input_ids"][0]
        ]

        return self

    @override
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.model.idx_tracker.batch_idx += 1

    def next(self) -> None:
        self.model.idx_tracker.generation_idx += 1

        self.model.run_graph("_", *self.args, **self.kwargs)
