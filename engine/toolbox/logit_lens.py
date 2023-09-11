import torch
from engine import Model, util
from engine.fx.Proxy import Proxy
from engine.Module import Module
import plotly.express as px
from typing import Callable

def logit_lens(
    model: Model,
    prompt: str,
    logit_fn: Callable,
    layers_module: Module,
    outpath: str,
    device_map="cpu",
):

    with model.generate(device_map=device_map, max_new_tokens=1) as generator:
        with generator.invoke(prompt) as invoker:
            tokens = invoker.tokens

            hidden_states = [
                layers_module[layer_idx].output[0]
                for layer_idx in range(len(layers_module))
            ]
    

    fig = px.imshow(
        ioi_patching_results,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Position", "y": "Layer"},
        x=token_labels,
        title="Normalized Logit Difference After Patching Residual Stream on the IOI Task",
    )

    fig.write_image(outpath)

    return ioi_patching_results
