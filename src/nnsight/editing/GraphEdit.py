from .Editor import Edit
import torch
from .. import util
from ..tracing.Graph import Graph
class GraphEdit(Edit):

    def __init__(self, module_path:str, graph:Graph) -> None:
        super().__init__()

        self.module_path = module_path
        self.graph = graph

        self.forward = None

    def edit(self, model:torch.nn.Module):
        module: torch.nn.Module = util.fetch_attr(model, self.module_path)
        self.forward = module.forward
        self.graph.wrap(module)

    def restore(self, model:torch.nn.Module):
        module: torch.nn.Module = util.fetch_attr(model, self.module_path)
        setattr(module, 'forward', self.forward)