from nnsight import LanguageModel

model = LanguageModel("gpt2", device_map='cuda:0')


def get_scores():
    hs = model.transformer.h[-1].output[0]

    return model.lm_head(model.transformer.ln_f(hs))


def decode(scores):
    scores = scores.argmax(dim=2)[0, -1]
    return model.tokenizer.decode(scores)


with model.generate(
    max_new_tokens=3,
    return_dict_in_generate=True,
    output_scores=True,
) as generator:
    with generator.invoke(
        "Madison square garden is located in the city of New"
    ) as invoker:
        tokenized = invoker.tokens

        # Reference the hidden states of the last layer for each token of the nine tokens (shape: (1,9,768))
        # Apply lm_head (decode into vocabulary space) and copy and return value (shape: (1,9,50257))
        logits1 = get_scores().save()

        # Denote that you are generating a token and subsequent interventions will apply to that generation
        # and not the previous ones.
        invoker.next()

        # Here the shape of the hidden states is (1, 1, 768) as there is just the one token
        # Get its hidden states of the last layer decoded as well
        logits2 = get_scores().save()

        # And again....
        invoker.next()

        logits3 = get_scores().save()


pred1 = decode(logits1.value)
pred2 = decode(logits2.value)
pred3 = decode(logits3.value)

print(pred1)
print(pred2)
print(pred3)
