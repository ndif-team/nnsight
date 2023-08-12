from .Promise import Proxy
import torch.fx


class Module:
    def __init__(self, mod_name) -> None:
        self._output = None
        self.output_shape = None
        self.output_type = torch.Tensor
        self.mod_name = mod_name

        super().__init__()

    @property
    def output(self):
        if self._output is None:
            node = Test.graph.placeholder(self.mod_name, type_expr=self.output_type)
            self._output = Proxy(node, Test.execute)
            node.graph.placeholders.append(self._output)
        return self._output

    @output.setter
    def output(self, value: Proxy):
        output = self.output
        Test.graph.call_function(output.set_result, args=(value.node,))
        Test.execute()


# use fututure collect all
class Test(torch.nn.Module):
    exe_graphs = []
    graph = torch.fx.graph.Graph()
    graph.placeholders = []

    @classmethod
    def execute(cls):
        Test.exe_graphs.append(Test.graph)
        Test.graph = torch.fx.graph.Graph()
        Test.graph.placeholders = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.m1 = Module("m1")
        self.m2 = Module("m2")


tm = Test()
m1 = tm.m1.output.save() + torch.Tensor([1, 2, 3])

tm.m2.output = m1
print(Test.exe_graphs[0])
print(Test.exe_graphs[1])
breakpoint()
