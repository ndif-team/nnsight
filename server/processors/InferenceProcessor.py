from typing import Dict

import accelerate
from engine import Intervention, Model, logger as engine_logger
from engine.modeling import JobStatus, RequestModel, ResponseModel

from ..ResponseDict import ResponseDict
from . import Processor


class InferenceProcessor(Processor):
    """
    Handles the LLM inference processing.

    Attributes
    ----------
        model_name_or_path : str
            repo id of hugging face LLM model repository or path to pre-cached checkpoint directory.
        device_map : Dict
            mapping of model modules to specific devices. To be used by accelerate if max_memory is None.
        max_memory : Dict[int,str]
            mapping of device to max allowed memory. To be used by accelerate to generate device_map.
        response_dict : ResponseDict
    """

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
        # Create Model
        self.model = Model(self.model_name_or_path)

        # If max_memory is set, use accelerate.infer_auto_device_map to get a device_map
        if self.max_memory is not None:
            self.model.graph.tie_weights()
            self.device_map = accelerate.infer_auto_device_map(
                self.model.graph, max_memory=self.max_memory
            )

        # Actually load the parameters of the model according to device_map
        self.model.dispatch(device_map=self.device_map)

        engine_logger.addHandler(self.logging_handler)

        super().initialize()

    def process(self, request: RequestModel) -> None:

        try:

            # Parse out data needed for inference
            execution_graphs, promises, prompts = (
                request.execution_graphs,
                request.promises,
                request.prompts,
            )
            # Promises are expected to be dictionary objects
            promises = {id: value.model_dump() for id, value in promises.items()}
            args, kwargs = request.args, request.kwargs

            # Run model with paramters and interventions
            output = self.model.run_model(
                execution_graphs, promises, prompts, *args, **kwargs
            )

            # Create response
            self.response_dict[request.id] = ResponseModel(
                id=request.id,
                recieved=request.recieved,
                blocking=request.blocking,
                status=JobStatus.COMPLETED,
                description="Your job has been completed.",
                output=output,
                # Move all copied data to cpu
                copies={
                    id: Intervention.Intervention.interventions[id].cpu().value
                    for id in Intervention.Copy.copies
                },
            ).log(self.logger)

            # Reset the model of all state data
            Model.clear()

        except Exception as exception:

            self.response_dict[request.id] = ResponseModel(
                id=request.id,
                recieved=request.recieved,
                blocking=request.blocking,
                status=JobStatus.ERROR,
                description=str(exception),
            ).log(self.logger)

            raise exception




