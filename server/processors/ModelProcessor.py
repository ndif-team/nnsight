from typing import Dict

import accelerate
from engine import Intervention, Model
from engine import logger as engine_logger
from engine.modeling import JobStatus, RequestModel, ResponseModel

from ..ResponseDict import ResponseDict
from . import Processor


class ModelProcessor(Processor):
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
            args, kwargs = request.args, request.kwargs

            tree = Intervention.InterventionTree.from_pydantic(request.interventions)

            # Run model with paramters and interventions
            output = self.model.run_model(tree, request.prompts, *args, **kwargs)

            # Create response
            self.response_dict[request.id] = ResponseModel(
                id=request.id,
                recieved=request.recieved,
                blocking=request.blocking,
                status=JobStatus.COMPLETED,
                description="Your job has been completed.",
                output=output,
                # Move all copied data to cpu
                saves={
                    name: tree.save_interventions[name].value().cpu()
                    for name in tree.save_interventions
                },
            ).log(self.logger)

        except Exception as exception:
            self.response_dict[request.id] = ResponseModel(
                id=request.id,
                recieved=request.recieved,
                blocking=request.blocking,
                status=JobStatus.ERROR,
                description=str(exception),
            ).log(self.logger)

            raise exception
