from nnsight import LanguageModel
m = LanguageModel("gpt2", device_map="cpu", dispatch=True)
with m.trace("Hello world"):
    h6 = m.transformer.h[6].output[0].save()
    logits = m.lm_head.output.save()
print("h6", tuple(h6.shape), "logits", tuple(logits.shape), "SMOKE_OK")
