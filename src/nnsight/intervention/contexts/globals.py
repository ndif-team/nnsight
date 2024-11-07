from __future__ import annotations

from inspect import getmembers, isclass

import torch
from torch.utils import data

from ...tracing.contexts.globals import global_patch, GlobalTracingContext, global_patch_fn
from . import InterventionTracer

global_patch(torch.nn.Parameter),
global_patch(data.DataLoader),
global_patch(torch.arange),
global_patch(torch.empty),
global_patch(torch.eye),
global_patch(torch.full),
global_patch(torch.linspace),
global_patch(torch.logspace),
global_patch(torch.ones),
global_patch(torch.rand),
global_patch(torch.randint),
global_patch(torch.randn),
global_patch(torch.randperm),
global_patch(torch.zeros),
global_patch(torch.cat),

for key, value in getmembers(torch.optim, isclass):

    if issubclass(value, torch.optim.Optimizer):

        global_patch(value)


class GlobalInterventionTracingContext(GlobalTracingContext, InterventionTracer):
    GLOBAL_TRACING_CONTEXT: GlobalInterventionTracingContext

GlobalTracingContext.GLOBAL_TRACING_CONTEXT = GlobalInterventionTracingContext()
