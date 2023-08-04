from engine import Model
  
model = Model('gpt2')


def get_scores():

    hs = model.transformer.h[-1].output[0]

    return model.lm_head(model.transformer.ln_f(hs))

def decode(scores):

    scores = scores.argmax(dim=2)[0, -1]
    return model.tokenizer.decode(scores)

with model.invoke('Madison square garden is located in the city of New') as invoker:

    tokenized = invoker.tokens
    
    # Reference the hidden states of the last layer for each token of the nine tokens (shape: (1,9,768))
    # Apply lm_head (decode into vocabulary space) and copy and return value (shape: (1,9,50257))    
    logits1 = get_scores().copy()

    # Denote that you are generating a token and subsequent interventions will apply to that generation
    # and not the previous ones. 
    invoker.next()

    # Here the shape of the hidden states is (1, 1, 768) as there is just the one token
    # Get its hidden states of the last layer decoded as well
    logits2 = get_scores().copy()

    # And again....
    invoker.next()

    logits3 = get_scores().copy()

output = model(device='server', max_new_tokens=3, return_dict_in_generate=True, output_scores=True)

pred1 = decode(logits1.value)
pred2 = decode(logits2.value)
pred3 = decode(logits3.value)
breakpoint()
