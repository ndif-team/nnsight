from typing import Any

from ...contexts import Runner


class GenerationMixin:

    def generate(self, *args, **kwargs) -> Runner:
        """The ``.generate(...)`` context is meant for multi-iteration runs. Arguments passed to generate determine the generation behavior — in this case to generate three tokens.
        Within a generation context, invoker sub-contexts are entered using ``generator.invoke``. This is where an input (or batch of inputs) to the model is accepted, and batched with other invocations. It's in these contexts where operations on inputs and outputs of modules are tracked and prepared for execution.

        In this example, we run two prompts on the language model in order to generate two tokens. We also perform a ``.save()`` operation on the output of the lm_head module (the logit outputs) in order to save these activations and access them after generation is over:

        .. code-block:: python

        with model.generate(max_new_tokens=2) as generator:
            with generator.invoke("The Eiffel Tower is in the city of") as invoker:
                logits11 = model.lm_head.output.save()
                logits12 = model.lm_head.next().output.save()
            with generator.invoke("The Empire State Building is in the city of") as invoker:
                logits21 = model.lm_head.output.save()
                logits22 = model.lm_head.next().output.save()

        print(logits1.value)
        print(logits2.value)

        Returns:
            _type_: _description_
        """

        return Runner(self, *args, generate=True, **kwargs)

    def _execute(
        self, prepared_inputs: Any, *args, generate: bool = False, **kwargs
    ) -> Any:

        if generate:

            return self._execute_generate(prepared_inputs, *args, **kwargs)

        return self._execute_forward(prepared_inputs, *args, **kwargs)

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
