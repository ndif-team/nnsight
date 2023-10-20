from nnsight import LanguageModel

model = LanguageModel("gpt2")

print(model)

with model.generate(server=True) as generator:
    with generator.invoke("Hello world") as invoker:
        hiddenstates = model.transformer.h[2].output.save()


print(hiddenstates.value)
