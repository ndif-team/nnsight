from nnsight import LanguageModel

model = LanguageModel("gpt2", device_map="cuda:0")


with model.invoke("hello world") as invoker:
    hidden_states = model.transformer.h[-1].output[0].save()


print(invoker.output)
print(hidden_states.value)
