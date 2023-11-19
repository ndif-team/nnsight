import torch

class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n, n)
        # we don't need init if the saved checkpoint has a nice
        # starting point already.
        # you can also study this if you want, but it is our focus.
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)

