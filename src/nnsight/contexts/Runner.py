from __future__ import annotations

from .Tracer import Tracer


class Runner(Tracer):
    def __init__(
        self,
        input,
        *args,
        inference:bool=True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input = input
        self.inference = inference

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
