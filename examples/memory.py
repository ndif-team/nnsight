from nnsight import LanguageModel
import objgraph
import random
import gc
import torch
model = LanguageModel('EleutherAI/pythia-70m', device_map='cuda:0')



i = 0

with torch.no_grad():
    while True:

        with model.forward("sadas sadasjasd", scan=False):
            pass
