from nnsight import LanguageModel
import torch

model = LanguageModel("gpt2", device_map='cuda:0')

with model.generate() as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:
        
        hidden_states = model.transformer.h[-1].output[0]
        hidden_states = model.lm_head(model.transformer.ln_f(hidden_states)).save()
        tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2).save()
        
print(hidden_states.value)
print(tokens.value)
print(model.tokenizer.decode(tokens.value[0]))
