from .NNsight import NNsight
from .Envoy import Envoy

import torch


class SM(torch.nn.Module):
    
    
    def forward(self, x):
        
        return x + 1

class MM(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.sm = SM()
    
    def forward(self, x):
        
        x = self.sm(x)
        
        return x * 2


model = Envoy(MM())

with model.trace(10):
    
    for i in range(10):
        
        print(model.sm.output + i)
        
        if i == 5:
            model.sm.output = 99999999
        
    zzz = torch.tensor(10)
    
    out = model.output
    
    with model.trace(20):
        
        print(out * 100)
    
    
with model.swag():
    print(out * 100)
    
print(out * 100)
    
