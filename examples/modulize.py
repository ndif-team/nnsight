from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda:0')

print(model.transformer.h[1].attn.graph)

model.modulize(model.transformer.h[1].attn, 'softmax_0', 'attention_probs')

with model.generate(max_new_tokens=3) as generator:
    
    with generator.invoke('Hello world') as invoker:

        attention_probs = model.transformer.h[1].attn.attention_probs.output.save()
    
print(attention_probs.value)

