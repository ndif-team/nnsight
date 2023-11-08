from nnsight import LanguageModel
from nnsight.tracing.Proxy import Proxy
from nnsight import util
import torch

model = LanguageModel('gpt2', device_map='cuda:0')

clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

correct_index = model.tokenizer(" John")['input_ids'][0]
incorrect_index = model.tokenizer(" Mary")['input_ids'][0]


with model.generate(max_new_tokens=1) as generator:

    with generator.invoke(clean_prompt) as invoker:

        clean_tokens = invoker.input['input_ids'][0]

        clean_hs = [model.transformer.h[layer_idx].output[0] for layer_idx in range(len(model.transformer.h))]

        clean_logits = model.lm_head.output

        clean_logit_diff = (clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]).save()

    with generator.invoke(corrupted_prompt) as invoker:

        corrupted_logits = model.lm_head.output

        corrupted_logit_diff = (corrupted_logits[0, -1, correct_index] - corrupted_logits[0, -1, incorrect_index]).save()

    ioi_patching_results = []

    for layer_idx in range(len(model.transformer.h)):

        _ioi_patching_results = []

        for token_idx in range(len(clean_tokens)):

            with generator.invoke(corrupted_prompt) as invoker:

                model.transformer.h[layer_idx].output[0].t[token_idx] = clean_hs[layer_idx].t[token_idx]

                patched_logits = model.lm_head.output

                patched_logit_diff = patched_logits[0, -1, correct_index] - patched_logits[0, -1, incorrect_index]

                patched_result = (patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff)

                _ioi_patching_results.append(patched_result.save())

        ioi_patching_results.append(_ioi_patching_results)


print(f"Clean logit difference: {clean_logit_diff.value:.3f}")
print(f"Corrupted logit difference: {corrupted_logit_diff.value:.3f}")
breakpoint()
ioi_patching_results = util.apply(ioi_patching_results, lambda x : x.value, Proxy)
ioi_patching_results = util.apply(ioi_patching_results, lambda x : x.item(), torch.Tensor)

import plotly.express as px
clean_tokens = [model.tokenizer.decode(token) for token in clean_tokens]
token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]

fig = px.imshow(ioi_patching_results, color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":"Position", "y":"Layer"}, x=token_labels, title="Normalized Logit Difference After Patching Residual Stream on the IOI Task")

fig.write_image("patching.png")

breakpoint()