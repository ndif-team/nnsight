from typing import Any

from ...contexts import Runner


class GenerationMixin:

    def generate(self, *args, **kwargs):

        Runner(self, *args, generate=True, **kwargs)

    def _execute(self, prepared_inputs: Any, *args, generate=False, **kwargs) -> Any:

        if generate:

            return self._execute_generate(prepared_inputs, *args, **kwargs)

        return self._execute_forward(prepared_inputs, *args, **kwargs)

    def _execute_forward(self, prepared_inputs: Any, *args, **kwargs):

        raise NotImplementedError()

    def _execute_generate(self, prepared_inputs: Any, *args, **kwargs):

        raise NotImplementedError()
