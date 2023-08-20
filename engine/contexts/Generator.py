from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Dict, List

import socketio
import torch.fx

from .. import CONFIG, logger, modeling
from ..fx.Tracer import Tracer
from ..Intervention import InterventionTree
from .Invoker import Invoker

if TYPE_CHECKING:
    from ..Model import Model


class Generator:
    def __init__(
        self, model: "Model", *args, device_map="server", blocking=True, **kwargs
    ) -> None:
        self.model = model
        self.device_map = device_map
        self.blocking = blocking
        self.args = args
        self.kwargs = kwargs

        self.generation_idx: int = 0
        self.batch_idx: int = 0
        self.prompts: List = []
        self.tracer: Tracer = Tracer(
            torch.fx.graph.Graph(owning_module=self.model.meta_model)
        )
        self.output = None

        for name, module in self.model.meta_model.named_modules():
            module.generator = self

    def __enter__(self) -> Generator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.tracer.graph.eliminate_dead_code()

        interventions = modeling.InterventionModel.from_graph(self.tracer.graph)

        if self.device_map == "server":
            self.run_server(interventions)
        else:
            self.run_local(interventions)

    def run_local(self, interventions: Dict[str, modeling.InterventionModel]):
        tree = InterventionTree.from_pydantic(interventions)

        self.output = self.model(self.prompts, tree, *self.args, **self.kwargs)

        for name in self.tracer.save_proxies:
            self.tracer.save_proxies[name].set_result(tree.interventions[name].value())

    def run_server(self, interventions: Dict[str, modeling.InterventionModel]):
        request = modeling.RequestModel(
            args=self.args,
            kwargs=self.kwargs,
            model_name=self.model.model_name_or_path,
            prompts=self.prompts,
            interventions=interventions,
        )

        if self.blocking:
            self.blocking_request(request)
        else:
            self.non_blocking_request(request)

    def blocking_request(self, request: modeling.RequestModel):
        sio = socketio.Client()
        sio.connect(f"ws://{CONFIG.API.HOST}")

        @sio.on("blocking_response")
        def blocking_response(data):
            data: modeling.ResponseModel = pickle.loads(data)

            logger.info(str(data))

            if data.status == modeling.JobStatus.COMPLETED:
                for name, value in data.saves.items():
                    self.tracer.save_proxies[name].set_result(value)

                self.output = data.output

                sio.disconnect()

            elif data.status == modeling.JobStatus.ERROR:
                sio.disconnect()

        sio.emit("blocking_request", pickle.dumps(request))

        sio.wait()

    def non_blocking_request(self, request: modeling.RequestModel):
        pass

    def invoke(self, input) -> Invoker:
        return Invoker(self, input)
