from nnsight import LanguageModel
from nnsight.tracing.Proxy import Proxy
from nnsight import util

model = LanguageModel('gpt2', device_map='cuda:0')

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."

with model.generate(max_new_tokens=1, output_attentions=True) as generator:

    with generator.invoke(gpt2_text, output_attentions=True) as invoker:

        tokens = invoker.tokens

        attn_hidden_states = [model.transformer.h[layer_idx].attn.output[2][0].save() for layer_idx in range(len(model.transformer.h))]

attn_hidden_states = util.apply(attn_hidden_states, lambda x : x.value, Proxy)

import circuitsvis as cv

cv.attention.attention_patterns(tokens=tokens, attention=attn_hidden_states[0])
