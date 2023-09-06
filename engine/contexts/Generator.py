from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Dict, Union

import socketio

from .. import CONFIG, modeling
from ..fx.Graph import Graph
from ..intervention import InterventionProxy
from .Invoker import Invoker

if TYPE_CHECKING:
    from ..Model import Model


class Generator:
    """_summary_

    Attributes:
        model (Model): Model object this is a generator for.
        device_map (Union[str,Dict]): What device/device map to run the model on. Defaults to 'server'
        blocking (bool): If when using device_map='server', block and wait form responses. Otherwise have to manually
            request a response.
        args (List[Any]): Arguments for calling the model.
        kwargs (Dict[str,Any]): Keyword arguments for calling the model.
        generation_idx (int): Keeps track of what iteration of generation to do interventions at. Used by the Module class
            to specify generation_idx for interventions and changed by the Invoker class using invoker.next().
        batch_idx (int): Keeps track of which batch in generation to do interventions at. Used by the Module class
            to specify batch_idx for interventions and changed by the Invoker class using invoker.__exit__().
        prompts (List[str]): Keeps track of prompts used by invokers.
        graph (Graph): Graph of all user intervention operations.
        output (??): desc
    """
    def __init__(
        self, model: "Model", *args, device_map:Union[str,Dict]="server", blocking:bool=True, **kwargs
    ) -> None:
        
        self.model = model
        self.device_map = device_map
        self.blocking = blocking
        self.args = args
        self.kwargs = kwargs

        self.generation_idx: int = 0
        self.batch_idx: int = 0
        self.prompts: List[str] = []
        self.graph = Graph(self.model.meta_model, proxy_class=InterventionProxy)
        self.output = None

        # Modules need to know about the current generator to create the correct proxies.
        for name, module in self.model.meta_model.named_modules():
            module.generator = self

    def __enter__(self) -> Generator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """On exit, run and generate using the model whether locally or on the server.  
        """
        if self.device_map == "server":
            self.run_server()
        else:
            self.run_local()

    def run_local(self):

        # Dispatch the model to the correct device.
        self.model.dispatch(device_map=self.device_map)

        # Run the model and store the output.
        self.output = self.model(self.prompts, self.graph, *self.args, **self.kwargs)

    def run_server(self):

        # Create the pydantic class for the request.
        request = modeling.RequestModel(
            args=self.args,
            kwargs=self.kwargs,
            model_name=self.model.model_name_or_path,
            prompts=self.prompts,
            # Convert Graph class into transferable json format
            intervention_graph=modeling.fx.NodeModel.from_graph(self.graph),
        )

        if self.blocking:
            self.blocking_request(request)
        else:
            self.non_blocking_request(request)

    def blocking_request(self, request: modeling.RequestModel):

        # Create a socketio connection to the server.
        sio = socketio.Client()
        sio.connect(f"ws://{CONFIG.API.HOST}")

        # Called when recieving a response from the server.
        @sio.on("blocking_response")
        def blocking_response(data):

            # Load the data into the ResponseModel pydantic class.
            data: modeling.ResponseModel = pickle.loads(data)

            # Print response for user ( should be logger.info and have an infor handler print to stdout)
            print(str(data))

            # If the status of the response is completed, update the local futures that the user specified to save. 
            # Then disconnect and continue.
            if data.status == modeling.JobStatus.COMPLETED:
                for name, value in data.saves.items():
                    self.graph.nodes[name].future.set_result(value)

                self.output = data.output

                sio.disconnect()
            # Or if there was some error.
            elif data.status == modeling.JobStatus.ERROR:
                sio.disconnect()

        sio.emit("blocking_request", pickle.dumps(request))

        sio.wait()

    def non_blocking_request(self, request: modeling.RequestModel):
        pass

    def invoke(self, input) -> Invoker:
        return Invoker(self, input)
