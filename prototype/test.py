from .models import llm, NDIFInvoker

model, tokenizer = llm('gpt2')

input_ids = tokenizer("Hello world", return_tensors='pt')["input_ids"]

with NDIFInvoker(model, input_ids) as invoker:

    mzzz = model.h[0].mlp.output

    mmlp = mzzz + model.h[1].mlp.output

    model.h[0].mlp.output = mmlp
