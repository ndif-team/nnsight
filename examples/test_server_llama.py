from nnsight import LanguageModel

model = LanguageModel('decapoda-research/llama-65b-hf')

print(model)

with model.generate(server=True, max_new_tokens=1, return_dict_in_generate=True, output_scores=True) as generator:
    with generator.invoke('Hello world') as invoker:
        
        hiddenstates = model.model.layers[0].output.save()

print(hiddenstates.value)
