from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List

import socketio

from .. import CONFIG, modeling
from ..fx.Graph import Graph
from ..intervention import InterventionProxy
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
        self.graph = Graph(self.model.meta_model, proxy_class=InterventionProxy)
        self.output = None

        for name, module in self.model.meta_model.named_modules():
            module.generator = self

    def __enter__(self) -> Generator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.device_map == "server":
            self.run_server()
        else:
            self.run_local()

    def run_local(self):
        self.model.dispatch(device_map=self.device_map)

        self.output = self.model(self.prompts, self.graph, *self.args, **self.kwargs)

    def run_server(self):
        request = modeling.RequestModel(
            args=self.args,
            kwargs=self.kwargs,
            model_name=self.model.model_name_or_path,
            prompts=self.prompts,
            intervention_graph=modeling.fx.NodeModel.from_graph(self.graph),
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

            print(str(data))

            if data.status == modeling.JobStatus.COMPLETED:
                for name, value in data.saves.items():
                    self.graph.nodes[name].future.set_result(value)

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
