from __future__ import annotations

from inspect import getmembers, isclass

import torch
from torch.utils import data

from ...tracing.contexts.globals import (GlobalTracingContext, global_patch,
                                         global_patch_fn)
from . import InterventionTracer

global_patch(torch.nn.Parameter)
global_patch(data.DataLoader)
global_patch(torch.arange)
global_patch(torch.empty)
global_patch(torch.eye)
global_patch(torch.full)
global_patch(torch.linspace)
global_patch(torch.logspace)
global_patch(torch.ones)
global_patch(torch.rand)
global_patch(torch.randint)
global_patch(torch.randn)
global_patch(torch.randperm)
global_patch(torch.zeros)
global_patch(torch.cat)

for key, value in getmembers(torch.optim, isclass):

    if issubclass(value, torch.optim.Optimizer):

        global_patch(value)
        
import math
from inspect import getmembers, isbuiltin, isfunction

import einops

for key, value in getmembers(einops.einops, isfunction):
    global_patch(value)
for key, value in getmembers(math, isbuiltin):
    global_patch(value)


class GlobalInterventionTracingContext(GlobalTracingContext, InterventionTracer):
    GLOBAL_TRACING_CONTEXT: GlobalInterventionTracingContext

GlobalTracingContext.GLOBAL_TRACING_CONTEXT = GlobalInterventionTracingContext()
