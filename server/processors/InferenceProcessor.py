from engine import Intervention, Model
from engine.models import JobStatus, RequestModel, ResponseModel

from ..ResponseDict import ResponseDict
from . import Processor


class InferenceProcessor(Processor):
    def __init__(
        self, model_name_or_path: str, response_dict: ResponseDict, *args, **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.response_dict = response_dict

        super().__init__(*args, **kwargs)

    def initialize(self):
        self.model = Model(self.model_name_or_path, dispatch=True)

        super().initialize()

    def process(self, request: RequestModel):
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
                id: Intervention.Intervention.interventions[id]._value
                for id in Intervention.Copy.copies
            },
        )

        Model.clear()

        self.response_dict[response.id] = response
