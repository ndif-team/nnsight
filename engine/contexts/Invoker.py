from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
    from .Generator import Generator


class Invoker:
    def __init__(self, generator: "Generator", input, *args, **kwargs) -> None:
        self.generator = generator
        self.input = input
        self.args = args
        self.kwargs = kwargs
        self.tokens = None

    @override
    def __enter__(self) -> Invoker:
        # Were in a new invocation so set generation_idx to 0
        self.generator.generation_idx = 0

        # Run graph_mode with meta tensors to collect shape information
        inputs = self.generator.model.prepare_inputs(self.input)
        self.generator.model.run_graph(inputs.copy(), *self.args, **self.kwargs)

        # Decode tokenized inputs for use usage
        self.tokens = [
            self.generator.model.tokenizer.decode(token)
            for token in inputs["input_ids"][0]
        ]

        # Rebuild prompt from tokens (do this becuase if they input ids directly, we still need to pass
        # all input data at once to a tokenizer to correctly batch the attention)
        self.generator.prompts.append("".join(self.tokens))

        return self

    @override
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Exiting an invocation so if we enter a new one, it will be a new batch idx
        self.generator.batch_idx += 1

    def next(self) -> None:
        # .next() increases which generation idx the interventions happen
        self.generator.generation_idx += 1

        # Run graph with
        inputs = self.generator.model.prepare_inputs("_")
        self.generator.model.run_graph(inputs, *self.args, **self.kwargs)
