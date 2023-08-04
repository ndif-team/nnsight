from typing import Dict

import accelerate
from engine import Intervention, Model
from engine.modeling import JobStatus, RequestModel, ResponseModel
from huggingface_hub import try_to_load_from_cache
from ..ResponseDict import ResponseDict
from . import Processor


class InferenceProcessor(Processor):
    def __init__(
        self,
        model_name_or_path: str,
        device_map: Dict,
        max_memory: Dict[int, str],
        response_dict: ResponseDict,
        *args,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.max_memory = max_memory
        self.device_map = device_map
        self.response_dict = response_dict

        super().__init__(*args, **kwargs)

    def initialize(self) -> None:
        self.model = Model(self.model_name_or_path)

        if self.max_memory is not None:

            self.model.graph.tie_weights()
            self.device_map = accelerate.infer_auto_device_map(
                self.model.graph, max_memory=self.max_memory
            )

        self.model.dispatch(device_map=self.device_map)

        super().initialize()

    def process(self, request: RequestModel) -> None:
        execution_graphs, promises, prompts = (
            request.execution_graphs,
            request.promises,
            request.prompts,
        )
        promises = {id: value.model_dump() for id, value in promises.items()}
        args, kwargs = request.args, request.kwargs

        output = self.model.run_model(
            execution_graphs, promises, prompts, *args, **kwargs
        )

        response = ResponseModel(
            id=request.id,
            blocking=request.blocking,
            status=JobStatus.COMPLETED,
            description="Your job has been completed.",
            output=output,
            copies={
                id: Intervention.Intervention.interventions[id].cpu().value
                for id in Intervention.Copy.copies
            },
        )

        Model.clear()

        self.response_dict[response.id] = response
