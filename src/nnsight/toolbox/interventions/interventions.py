import torch
from abc import ABC, abstractmethod

from transformations import RotateLayer
from interventions_utils import sigmoid_boundary


class Intervention(torch.nn.Module, ABC):

    """Intervention the original representations."""
    def __init__(self):
        super().__init__()
        self.trainble = False
        
    @abstractmethod
    def set_interchange_dim(self, interchange_dim):
        pass

    @abstractmethod
    def forward(self, base, source):
        pass

class TrainbleIntervention(Intervention):

    """Intervention the original representations."""
    def __init__(self):
        super().__init__()
        self.trainble = True


class BoundlessRotatedSpaceIntervention(TrainbleIntervention):
    
    """Intervention in the rotated space with boundary mask."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        rotate_layer = RotateLayer(embed_dim)

        # Orthogonal parametrizations constrains the weights of the rotate layer to be 
        # orthogonal during training
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer)

        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.tensor(50.0)) 
        self.embed_dim = embed_dim
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False)
        
    def get_boundary_parameters(self):
        return self.intervention_boundaries

    # temporarily in here
    def zero_grad(self):
        """Zero out the gradients of all parameters in this module."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp
        
    def set_interchange_dim(self, interchange_dim):
        """interchange dim is learned and can not be set"""
        assert False

    def forward(self, base, source):
        batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = torch.clamp(
            self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, 1), 
            0.,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature
        )

        # print(boundary_mask.get_device())
        boundary_mask = torch.ones(
            batch_size).unsqueeze(dim=-1).to(boundary_mask.device)*boundary_mask
        # boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (1. - boundary_mask)*rotated_base + boundary_mask*rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        # return output.to(base.dtype)
        return output
    
    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention(embed_dim={self.embed_dim})"