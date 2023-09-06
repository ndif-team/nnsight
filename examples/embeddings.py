from engine import Model

model = Model('gpt2')

print(model)

with model.generate(device_map='cuda:0', max_new_tokens=3) as generator:
    
    with generator.invoke("Madison square garden is located in the city of New") as invoker:

        embeddings = model.transformer.wte.output.save()

print(model.tokenizer.decode(generator.output[0]))
print(embeddings.value)

with model.generate(device_map='cuda:0', max_new_tokens=3) as generator:
    
    with generator.invoke("doo doo doo doo dosdfgsdo") as invoker:

        model.transformer.wte.output = embeddings.value

print(model.tokenizer.decode(generator.output[0]))
