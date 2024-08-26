from typing import Any

from ... import NNsight
from ...contexts.Tracer import Tracer

class GenerationMixin(NNsight):

    def generate(self, *args, **kwargs) -> Tracer:

        return self.trace(*args, generate=True, **kwargs)

    def _execute(
        self, *args, generate: bool = False, **kwargs
    ) -> Any:

        if generate:

            return self._execute_generate(*args, **kwargs)

        return self._execute_forward(*args, **kwargs)

    def _scan(
        self, prepared_inputs: Any, *args, generate: bool = False, **kwargs
    ) -> Any:

        if generate:

            return self._scan_generate(prepared_inputs, *args, **kwargs)

        return self._scan_forward(prepared_inputs, *args, **kwargs)

    def _execute_forward(self, prepared_inputs: Any, *args, **kwargs):

        raise NotImplementedError()

    def _execute_generate(self, prepared_inputs: Any, *args, **kwargs):

        raise NotImplementedError()

    def _scan_forward(self, prepared_inputs: Any, *args, **kwargs):

        raise NotImplementedError()

    def _scan_generate(self, prepared_inputs: Any, *args, **kwargs):

        raise NotImplementedError()
