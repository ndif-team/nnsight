from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from ..tracing.Proxy import Proxy
from .Tracer import Tracer


class Invoker:
    def __init__(
        self,
        tracer: Tracer,
        input,
        *args,
        scan: bool = True,
        **kwargs,
    ) -> None:
        self.tracer = tracer
        self.input = input
        self.scan = scan
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> Invoker:
        # Were in a new invocation so set generation_idx to 0,
        self.tracer.generation_idx = 0

        self.input = self.tracer.model._prepare_inputs(
            self.input, *self.args, **self.kwargs
        )

        if self.scan:
            self.tracer.model._scan(self.input, *self.args, **self.kwargs)
        else:
            for name, module in self.tracer.model.meta_model.named_modules():
                module._output = None
                module._input = None

        batched_inputs = self.tracer.model._batched_inputs(self.input)

        self.tracer.batch_size = len(batched_inputs)
        self.tracer.batched_input.extend(batched_inputs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def next(self, increment: int = 1) -> None:
        # .next() increases which generation idx the interventions happen.
        self.tracer.generation_idx += increment

        if self.scan:
            # Run graph with singe token input.
            self.inputs = self.tracer.model._prepare_inputs(
                self.tracer.model._example_input(), *self.args, **self.kwargs
            )
            self.tracer.model._scan(self.inputs, *self.args, **self.kwargs)
        else:
            for name, module in self.tracer.model.meta_model.named_modules():
                module._output = None
                module._input = None

    def save_all(self) -> Dict[str, Proxy]:
        """Saves the output of all modules and returns a dictionary of [module_path -> save proxy]

        Returns:
            Dict[str, Proxy]: _description_
        """
        result = {}

        for name, module in self.tracer.model.meta_model.named_modules():
            result[module.module_path] = module.output.save()

        return result
