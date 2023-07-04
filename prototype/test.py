from .models import llm, NDIFInvoker

model, tokenizer = llm('gpt2')

input_ids = tokenizer("Hello world", return_tensors='pt')["input_ids"]

with NDIFInvoker(model, input_ids) as invoker:

    m0mlp = model.h[0].mlp.output
    m1mlp = model.h[1].mlp.output

    mmlp = m0mlp + m1mlp

    model.h[0].mlp.output = mmlp
    
breakpoint()