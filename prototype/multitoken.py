from engine import Model
  
model = Model('gpt2')

with model.invoke('Madison square garden is located in the city of New') as invoker:

    tokenized = invoker.tokens
    
    # Reference the hidden states of the last layer for each token of the nine tokens (shape: (1,9, 768))
    # Apply lm_head (decode into vocabulary space) and copy and return value (shape: (1,9, 50257))    
    hs1 = model.transformer.h[-1].output[0]
    value1 = model.lm_head(model.transformer.ln_f(hs1)).copy()

    # Denote that you are generating a token and subsequent interventions will apply to that generation
    # and not the previous ones. 
    invoker.next()

    # Here the shape of the hidden states is (1, 1, 768) as there is just the one token
    hs2 = model.transformer.h[-1].output[0]
    # Get its hidden states of the last layer decoded as well
    value2 = model.lm_head(model.transformer.ln_f(hs2)).copy()

    # And again....
    invoker.next()

    hs3 = model.transformer.h[-1].output[0]
    value3 = model.lm_head(model.transformer.ln_f(hs3)).copy()

output = model(device='cuda:0', max_new_tokens=3, return_dict_in_generate=True, output_scores=True)

value1 = value1.value.argmax(dim=2)[0]
pred_1 = model.tokenizer.decode(value1)

value12 = value2.value.argmax(dim=2)[0, -1]
pred_2 = model.tokenizer.decode(value2)

value3 = value3.value.argmax(dim=2)[0, -1]
pred_3 = model.tokenizer.decode(value3)
breakpoint()
