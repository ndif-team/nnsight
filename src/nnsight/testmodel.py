from collections import OrderedDict

import torch

from . import NNsight

input_size = 5
hidden_dims = 10
output_size = 2



net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
)

net =  NNsight(net)

inputs = torch.rand((1, input_size))

with net.trace(inputs) as tracer:
        
    output1 = net.layer1.output.save()
    output2 = net.layer2.output.save()
    output11 = output1.clone()
    output11[:] = 10000
    net.layer1.output = output11
    output1 = net.layer1.output.save()
    tracer.apply(print, output1 * 2)
    for x in output1[0]:
        
        if output1.sum() > 10000:
            
            tracer.apply(print, output2)
            
    tracer.apply(print, output1 * 2)

