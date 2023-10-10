from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Union

from ..fx.Graph import Graph
from ..intervention import InterventionProxy

if TYPE_CHECKING:
    from ..models.AbstractModel import AbstractModel


# TODO make parent class for Runner and Generator as Module depends on attributes
class Runner:
    def __init__(
        self,
        model: "AbstractModel",
        input,
        *args,
        inference=False,
        **kwargs,
    ) -> None:
        self.model = model

        self.input = input
        self.inference = inference
        self.args = args
        self.kwargs = kwargs

        self.graph = Graph(self.model.meta_model, proxy_class=InterventionProxy)

        self.batch_size: int = 0
        self.prompts: List[str] = []
        self.generation_idx = 0
        self.output = None

        # Modules need to know about the current generator to create the correct proxies.
        for name, module in self.model.named_modules():
            module.generator = self

    def __enter__(self) -> Runner:
        token_ids = self.model._run_meta(self.input, *self.args, **self.kwargs)

        # Decode tokenized inputs for user usage.
        self.tokens = [
            [self.model.tokenizer.decode(token) for token in ids] for ids in token_ids
        ]
        self.ids = token_ids

        self.batch_size = len(self.ids)

        self.prompts.extend(["".join(tokens) for tokens in self.tokens])

        if len(self.tokens) == 1:
            self.tokens = self.tokens[0]
            self.ids = self.ids[0]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.output = self.model(
            self.model._run_local,
            self.input,
            self.graph,
            *self.args,
            inference=self.inference,
            **self.kwargs,
        )
