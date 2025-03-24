import pytest
import torch
import torch.nn as nn
import nnsight
from typing import Tuple

class KWOnlyModel(nn.Module):
    """A simple image processing model that applies separate transformations to RGB channels. 
    Only accepts kwargs in its forward method."""
    def __init__(self):
        super().__init__()
        
        self.red_transform = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.green_transform = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(), 
            nn.Linear(8, 1)
        )
        
        self.blue_transform = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, *, red=None, green=None, blue=None):
        """Process each color channel separately with different architectures"""
        if red is None:
            red = torch.zeros(1)
        if green is None:
            green = torch.zeros(1) 
        if blue is None:
            blue = torch.zeros(1)
            
        red_out = self.red_transform(red.unsqueeze(-1))
        green_out = self.green_transform(green.unsqueeze(-1))
        blue_out = self.blue_transform(blue.unsqueeze(-1))
        
        return red_out + green_out + blue_out

@pytest.fixture(autouse=True)
def model_clear(kwargs_only_model: KWOnlyModel):
    # clear the model before each test
    kwargs_only_model._clear()
    return kwargs_only_model

@pytest.fixture(scope="module")
def kwargs_only_model(device: str):
    """Fixture for a model that only accepts kwargs in its forward method."""
    # Define or load your kwargs-only model here
    return nnsight.NNsight(KWOnlyModel())

@pytest.fixture(scope="module")
def kwargs_only_model_input():
    red = torch.tensor(1, dtype=torch.float32)
    green = torch.tensor(2, dtype=torch.float32)
    blue = torch.tensor(3, dtype=torch.float32)
    return red, green, blue

def test_kwargs_only_model(kwargs_only_model: KWOnlyModel, kwargs_only_model_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    red, green, blue = kwargs_only_model_input

    with kwargs_only_model.trace(red=red, green=green, blue=blue):
        output = kwargs_only_model.output.save()

    assert output.shape == (1,)

def test_kwargs_only_model_missing_kwargs(kwargs_only_model: KWOnlyModel):
    
    # This model does have a valid default value for each kwarg, so it should not raise an error.
    with kwargs_only_model.trace():
        output = kwargs_only_model.output.save()

    assert output.shape == (1,1)

def test_kwargs_only_model_missing_kwargs_with_default_values(kwargs_only_model: KWOnlyModel):
    # This model does not have a valid default value for each kwarg, so it should raise an error.
    
    default_value = torch.tensor([1], dtype=torch.float32)
    with pytest.raises(nnsight.util.NNsightError):
        with kwargs_only_model.trace(default_value):
            output = kwargs_only_model.output.save()

@pytest.mark.skipif(True, reason="Known bug: 0d tensor as default arg to model.trace() causes 'len() of a 0-d tensor' error")
def test_0d_tensor_as_default_arg(kwargs_only_model: KWOnlyModel):
    """Test that documents the current bug with 0d tensors as default args."""
    zero_d_tensor = torch.tensor(5, dtype=torch.float32)  # A 0d tensor
    
    # This should work but currently fails with "len() of a 0-d tensor"
    with kwargs_only_model.trace(zero_d_tensor):
        output = kwargs_only_model.output.save()