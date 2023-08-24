from engine import Model
from engine import util
from engine.fx.Tracer import ModuleTracer
from engine.Module import Module
import torch
import torch.fx
import torch.jit
import inspect

# Get model wrapper for any model you can get with AutoConfig.from_pretrained(model_name)
model = Model("gpt2")
module = model.transformer.h[0].attn
with model.generate(device_map="cuda:0", max_new_tokens=3) as generator:
    with generator.invoke("Hello world") as invoker:
        signature = inspect.signature(module.forward)
        concrete_args = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        node_name_to_value = {
            k: torch.empty(module.input_shape[i], device='meta')
            for i, (k, v) in enumerate(signature.parameters.items())
            if v.default is inspect.Parameter.empty
        }
        tracer = ModuleTracer(param_shapes_constant=True, node_name_to_value=node_name_to_value)
        graph = tracer.trace(module, concrete_args=concrete_args)
#_________________________

        def get_node(name:str) -> torch.fx.node.Node:

            for node in graph.nodes:

                if node.name == name:
                    return node
                
            return None
        
        class WrapperModule(Module, torch.nn.Module):

            def __init__(self) -> None:
                torch.nn.Module.__init__(self)
                Module.__init__(self)

            def forward(self, x):

                return x

        node = get_node("matmul")

        setattr(module, "test", WrapperModule())

        with graph.inserting_after(node):

            graph.call_module("test", args=(node,))
        gm = torch.fx.graph_module.GraphModule(module, graph)
        breakpoint()
