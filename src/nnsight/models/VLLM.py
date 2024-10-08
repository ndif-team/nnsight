from typing import List, Optional, Tuple, Union

from ..util import WrapperModule
from .mixins import GenerationMixin

try:
    from vllm import LLM, RequestOutput
    from vllm.distributed import (destroy_distributed_environment,
                                  destroy_model_parallel,
                                  init_distributed_environment,
                                  initialize_model_parallel)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.loader import _initialize_model
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
            model (torch.nn.Module): Underlying model of the vLLM instance.
            llm_engine (vllm.LLM): vLLM inference engine instance.
        '''
        
        def __init__(self, model, llm_engine = None) -> None:

            super().__init__()

            self.model = model

            self.llm_engine = llm_engine

    def __init__(self, model_key: str, *args, **kwargs) -> None:

        super().__init__(model_key, *args, **kwargs)

    def _load(self, repo_id: str, **kwargs) -> VLLModel:

        if self._model is None:

            # no parallelism during initialization
            kwargs["tensor_parallel_size"] = 1
            kwargs["pipeline_parallel_size"] = 1

            # creating vLLM Engine args
            engine_args = EngineArgs(
                model=repo_id,
                **kwargs,
            )

            # creating the vllm engine configuration
            engine_config_dict = engine_args.create_engine_config().to_dict()

            # starting the distributed environment
            init_distributed_environment(
                engine_config_dict["parallel_config"].world_size, 
                0,
                'tcp://127.0.0.1:47303', 
                0, 
                backend="nccl"
            )

            # start tensor parallel group
            initialize_model_parallel(
                engine_config_dict["parallel_config"].tensor_parallel_size,
                engine_config_dict["parallel_config"].pipeline_parallel_size, 
                'nccl'
            )
            
            # initialize the model
            model = _initialize_model(
                        model_config=engine_config_dict["model_config"], 
                        load_config=engine_config_dict["load_config"],
                        lora_config=None,
                        cache_config=engine_config_dict["cache_config"],
                        scheduler_config=engine_config_dict["scheduler_config"]
            ) 

            return VLLM.VLLModel(model)
        else:

            # destroy the distributed environment created from the initial model initialization
            destroy_model_parallel()
            destroy_distributed_environment()

            if "tensor_parallel_size" in kwargs.keys():
                if kwargs["tensor_parallel_size"] > 1:
                    raise Exception("Tensor Parallelism currently not supported with nnsight.VLLM")

            llm = LLM(repo_id, **kwargs)

            model = llm.llm_engine.model_executor.driver_worker.model_runner.model

            return VLLM.VLLModel(model, llm)

    def _execute(self, prepared_inputs: Union[List[str], str], *args, generate=True, **kwargs) -> List[RequestOutput]:

        output = self._model.llm_engine.generate(prepared_inputs, *args, use_tqdm=False, **kwargs)

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
        if batched_inputs is None:

            return (prepared_inputs, )

        return (batched_inputs[0] + prepared_inputs, )
