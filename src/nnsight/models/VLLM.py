from typing import Any, List, Tuple, Union, Optional

from ..util import WrapperModule
from .mixins import GenerationMixin

try:
    from vllm import LLM, RequestOutput
except Exception as e:

    raise type(e)(
        "Install vllm in your environment to use it with NNsight. " + \
        "https://docs.vllm.ai/en/latest/getting_started/installation.html"
    ) from e

class VLLM(GenerationMixin):
    ''' NNsight wrapper to conduct interventions on a vLLM inference engine.

    .. code-block:: python
        from nnsight.models.VLLM import VLLM
        from vllm import SamplingParams

        model = VLLM("gpt2")

        prompt = ["The Eiffel Tower is in the city of"]
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, stop=["."])

        with model.trace(prompt, sampling_params=sampling_params) as tracer:
            model.model.transformer.h[8].output[-1][:] = 0

            outputs = model.output.save()

        for output in outputs.value:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    '''

    class VLLModel(WrapperModule):
        ''' Pytorch Wrapper for the vLLM engine to work seamlessly with NNsight.

        Attributes:
            llm (vllm.LLM): vLLM inference engine instance.
            model (torch.nn.Module): Underlying model of the vLLM instance.
        '''
        
        def __init__(self, *args, **kwargs) -> None:

            super().__init__()

            self.llm = LLM(*args, dtype="half", **kwargs)

            self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

    def __init__(self, model_key: str, *args, **kwargs) -> None:

        model_key = self._load(model_key, **kwargs)

        super().__init__(model_key, *args, **kwargs)

    def _load(self, repo_id: str, **kwargs) -> VLLModel:

        model = VLLM.VLLModel(model=repo_id, **kwargs)

        return model

    def _execute(self, prepared_inputs: Union[List[str], str], *args, generate=True, **kwargs) -> List[RequestOutput]:

        output = self._model.llm.generate(prepared_inputs, *args, use_tqdm=False, **kwargs)

        output = self._model(output)

        return output
    
    def _prepare_inputs(self, *inputs: Union[List[str], str]) -> Tuple[Tuple[List[str]], int]:
        if isinstance(inputs[0], list):
            return inputs, len(inputs[0])
        else:
            return ([inputs[0]],), 1
    
    def _batch_inputs(
        self, 
        batched_inputs: Optional[Tuple[List[str]]],
        prepared_inputs: List[str],
    ) -> Tuple[List[str]]:
        breakpoint()
        if batched_inputs is None:

            return (prepared_inputs, )

        return (batched_inputs[0] + prepared_inputs, )
