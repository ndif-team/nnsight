from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Dict, List, Union

import socketio

from .. import CONFIG, pydantics
from ..fx.Graph import Graph
from ..intervention import InterventionProxy
from .Invoker import Invoker

if TYPE_CHECKING:
    from ..models.AbstractModel import AbstractModel


class Generator:
    """_summary_

    Attributes:
        model (Model): Model object this is a generator for.
        blocking (bool): If when using device_map='server', block and wait form responses. Otherwise have to manually
            request a response.
        args (List[Any]): Arguments for calling the model.
        kwargs (Dict[str,Any]): Keyword arguments for calling the model.
        generation_idx (int): Keeps track of what iteration of generation to do interventions at. Used by the Module class
            to specify generation_idx for interventions and changed by the Invoker class using invoker.next().
        batch_size (int): Current size of invocation batch. To be used by Module node creation
        prompts (List[str]): Keeps track of prompts used by invokers.
        graph (Graph): Graph of all user intervention operations.
        output (??): desc
    """

    def __init__(
        self,
        model: "AbstractModel",
        *args,
        blocking: bool = True,
        server: bool = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.server = server
        self.blocking = blocking
        self.args = args
        self.kwargs = kwargs

        self.graph = Graph(self.model.meta_model, proxy_class=InterventionProxy)

        self.generation_idx: int = 0
        self.batch_size: int = 0
        self.prompts: List[str] = []
        self.output = None

        # Modules need to know about the current generator to create the correct proxies.
        for name, module in self.model.named_modules():
            module.generator = self

    def __enter__(self) -> Generator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """On exit, run and generate using the model whether locally or on the server."""
        if self.server:
            self.run_server()
        else:
            self.run_local()

    def run_local(self):
        # Run the model and store the output.
        self.output = self.model(self.model._generation, self.prompts, self.graph, *self.args, **self.kwargs)

    def run_server(self):
        # Create the pydantic class for the request.
        request = pydantics.RequestModel(
            args=self.args,
            kwargs=self.kwargs,
            model_name=self.model.model_name_or_path,
            prompts=self.prompts,
            intervention_graph=self.graph,
        )

        if self.blocking:
            self.blocking_request(request)
        else:
            self.non_blocking_request(request)

    def blocking_request(self, request: pydantics.RequestModel):
        # Create a socketio connection to the server.
        sio = socketio.Client()
        sio.connect(f"ws://{CONFIG.API.HOST}")

        # Called when recieving a response from the server.
        @sio.on("blocking_response")
        def blocking_response(data):
            # Load the data into the ResponseModel pydantic class.
            data: pydantics.ResponseModel = pickle.loads(data)

            # Print response for user ( should be logger.info and have an infor handler print to stdout)
            print(str(data))

            # If the status of the response is completed, update the local futures that the user specified to save.
            # Then disconnect and continue.
            if data.status == pydantics.JobStatus.COMPLETED:
                for name, value in data.saves.items():
                    self.graph.nodes[name].future.set_result(value)

                self.output = data.output

                sio.disconnect()
            # Or if there was some error.
            elif data.status == pydantics.JobStatus.ERROR:
                sio.disconnect()

        sio.emit(
            "blocking_request",
            request.model_dump(exclude_defaults=True, exclude_none=True),
        )

        sio.wait()

    def non_blocking_request(self, request: pydantics.RequestModel):
        pass

    def invoke(self, input, *args, **kwargs) -> Invoker:
        return Invoker(self, input, *args, **kwargs)
