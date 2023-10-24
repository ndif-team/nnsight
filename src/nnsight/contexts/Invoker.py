from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from ..fx.Proxy import Proxy

if TYPE_CHECKING:
    from .Generator import Generator


class Invoker:
    def __init__(self, generator: "Generator", input, *args, **kwargs) -> None:
        self.generator = generator
        self.input = input
        self.args = args
        self.kwargs = kwargs
        self.tokens = None
        self.ids = None

    def __enter__(self) -> Invoker:
        # Were in a new invocation so set generation_idx to 0,
        self.generator.generation_idx = 0

        # Run graph_mode with meta tensors to collect shape information,
        token_ids = self.generator.model._run_meta(self.input, *self.args, **self.kwargs)

        # Decode tokenized inputs for user usage.
        self.tokens = [
            [self.generator.model.tokenizer.decode(token) for token in ids]
            for ids in token_ids
        ]
        self.ids = token_ids

        self.generator.batch_size = len(self.ids)

        # Rebuild prompt from tokens (do this becuase if they input ids directly, we still need to pass
        # all input data at once to a tokenizer to correctly batch the attention).
        self.generator.prompts.extend(["".join(tokens) for tokens in self.tokens])

        if len(self.tokens) == 1:
            self.tokens = self.tokens[0]
            self.ids = self.ids[0]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def next(self) -> None:
        # .next() increases which generation idx the interventions happen.
        self.generator.generation_idx += 1

        # Run graph with singe token input.
        self.generator.model._run_meta("_", *self.args, **self.kwargs)

    def save_all(self) -> Dict[str, Proxy]:
        """Saves the output of all modules and returns a dictionary of [module_path -> save proxy]

        Returns:
            Dict[str, Proxy]: _description_
        """
        result = {}

        for name, module in self.generator.model.meta_model.named_modules():
            result[module.module_path] = module.output.save()

        return result
