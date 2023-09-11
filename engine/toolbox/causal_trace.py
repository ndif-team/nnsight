import torch
from engine import Model, util
from engine.fx.Proxy import Proxy
from engine.Module import Module
import plotly.express as px


def causal_trace(
    model: Model,
    clean_prompt: str,
    corrupted_prompt: str,
    correct_token: str,
    incorrect_token: str,
    logit_module: Module,
    layers_module: Module,
    outpath: str,
    device_map="cpu",
):
    correct_index = model.tokenizer(correct_token)["input_ids"][0]
    incorrect_index = model.tokenizer(incorrect_token)["input_ids"][0]

    with model.generate(device_map=device_map, max_new_tokens=1) as generator:
        with generator.invoke(clean_prompt) as invoker:
            clean_tokens = invoker.tokens

            clean_hs = [
                layers_module[layer_idx].output[0]
                for layer_idx in range(len(layers_module))
            ]

            clean_logits = logit_module.output

            clean_logit_diff = (
                clean_logits[0, -1, correct_index]
                - clean_logits[0, -1, incorrect_index]
            ).save()

        with generator.invoke(corrupted_prompt) as invoker:
            corrupted_logits = logit_module.output

            corrupted_logit_diff = (
                corrupted_logits[0, -1, correct_index]
                - corrupted_logits[0, -1, incorrect_index]
            ).save()

        ioi_patching_results = []

        for layer_idx in range(len(layers_module)):
            _ioi_patching_results = []

            for token_idx in range(len(clean_tokens)):
                with generator.invoke(corrupted_prompt) as invoker:
                    layers_module[layer_idx].output[0].t[token_idx] = clean_hs[
                        layer_idx
                    ].t[token_idx]

                    patched_logits = logit_module.output

                    patched_logit_diff = (
                        patched_logits[0, -1, correct_index]
                        - patched_logits[0, -1, incorrect_index]
                    )

                    patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                        clean_logit_diff - corrupted_logit_diff
                    )

                    _ioi_patching_results.append(patched_result.save())

            ioi_patching_results.append(_ioi_patching_results)

    print(f"Clean logit difference: {clean_logit_diff.value:.3f}")
    print(f"Corrupted logit difference: {corrupted_logit_diff.value:.3f}")

    ioi_patching_results = util.apply(ioi_patching_results, lambda x: x.value, Proxy)
    ioi_patching_results = util.apply(
        ioi_patching_results, lambda x: x.item(), torch.Tensor
    )

    token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]

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
