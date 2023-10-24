from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda:0')

with model.generate(max_new_tokens=3) as generator:
    
    with generator.invoke("Madison square garden is located in the city of New") as invoker:

        embeddings = model.transformer.wte.output

print(generator.output)

model = LanguageModel(model.local_model, tokenizer=model.tokenizer)

with model.generate(max_new_tokens=3) as generator:
    
    with generator.invoke("Madison square garden is located in the city of New") as invoker:

        embeddings = model.transformer.wte.output

print(generator.output)
