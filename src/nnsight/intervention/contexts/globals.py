"""
Global patching allows us to add un-traceable operations to nnsight by replacing them with ones that use the GLOBAL_TRACING_CONTEXT to add the operation to the current graph.
"""
from __future__ import annotations

from inspect import getmembers, isclass

import torch
from torch.utils import data

from ...tracing.contexts.globals import (GlobalTracingContext, global_patch,
                                         global_patch_fn)
from . import InterventionTracer



# Torch classes
global_patch(torch.nn.Parameter)
global_patch(torch.nn.Linear)

global_patch(data.DataLoader)
# Tensor creation operations
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

# All Optimizers
for key, value in getmembers(torch.optim, isclass):

    if issubclass(value, torch.optim.Optimizer):

        global_patch(value)
        
import math
from inspect import getmembers, isbuiltin, isfunction

import einops
# Einops
for key, value in getmembers(einops.einops, isfunction):
    global_patch(value)
# math
for key, value in getmembers(math, isbuiltin):
    global_patch(value)

# Give it InterventionTracer methods
class GlobalInterventionTracingContext(GlobalTracingContext, InterventionTracer):
    GLOBAL_TRACING_CONTEXT: GlobalInterventionTracingContext

GlobalTracingContext.GLOBAL_TRACING_CONTEXT = GlobalInterventionTracingContext()
