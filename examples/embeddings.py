from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda:0')

print(model)

with model.generate(max_new_tokens=3) as generator:
    
    with generator.invoke("Madison square garden is located in the city of New") as invoker:

        embeddings = model.transformer.wte.output

    with generator.invoke("_ _ _ _ _ _ _ _ _ _") as invoker:

        model.transformer.wte.output = embeddings

print(model.tokenizer.decode(generator.output[0]))
print(model.tokenizer.decode(generator.output[1]))