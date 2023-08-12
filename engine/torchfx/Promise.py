
import torch
import torch.fx
import torch.futures

class Proxy(torch.futures.Future, torch.fx.Proxy):

    exe_graph = []

    def __init__(self, node, execute):

        self.execute = execute

        torch.fx.Proxy.__init__(self, node)
        torch.futures.Future.__init__(self)

    def save(self):

        node = self.node.graph.call_function(self.set_result, args=(self.node,))
        graph = self.execute()
        return Proxy(node, self)

torch.fx.proxy.Proxy = Proxy