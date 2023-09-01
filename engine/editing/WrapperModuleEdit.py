import torch

from .. import util
from .Editor import Edit

class WrapperModule(torch.nn.Module):

    def forward(self, *args, **kwargs):

        if len(args) == 1:
            args = args[0]

        return args


class WrapperModuleEdit(Edit):
    def __init__(self, module_path: str, module_name: str) -> None:
        super().__init__()

        self.module_path = module_path
        self.module_name = module_name

        self.wrapper = WrapperModule()

    def edit(self, model: torch.nn.Module):
        module: torch.nn.Module = util.fetch_attr(model, self.module_path)
        setattr(module, self.module_name, self.wrapper)

    def restore(self, model: torch.nn.Module):
        module: torch.nn.Module = util.fetch_attr(model, self.module_path)
        delattr(module, self.module_name)
