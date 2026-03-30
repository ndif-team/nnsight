# nnsight-serve

A lightweight single-model server backed by vLLM. vLLM does the heavy lifting (continuous batching, paged KV cache, request scheduling); nnsight injects intervention hooks on top.

Users who only need one model get a persistent server with nnsight intervention support — no vLLM restart per request, no NDIF cluster setup.

## Install

```bash
pip install nnsight[serve]
```

This pulls in `vllm`, `fastapi`, and `uvicorn` alongside the core nnsight dependencies.

For development (editable install from source):

```bash
pip install -e ".[serve]"
```

## Start the server

```bash
# Single GPU
nnsight-serve Qwen/Qwen3-30B-A3B --port 6677 --gpu-memory-utilization 0.8

# Multi-GPU (tensor parallel)
nnsight-serve meta-llama/Llama-3.1-70B --port 6677 --tensor-parallel-size 4

# With API key authentication
nnsight-serve Qwen/Qwen3-30B-A3B --port 6677 --api-key mysecret
```

All flags after the model name are forwarded to vLLM's engine (e.g., `--tensor-parallel-size`, `--gpu-memory-utilization`, `--enforce-eager`).

Default bind address is `127.0.0.1` (localhost only). Pass `--host 0.0.0.0` for network access.

If the `nnsight-serve` command is not found (common with editable installs), use:

```bash
python -m nnsight.modeling.vllm.serve.cli Qwen/Qwen3-30B-A3B --port 6677
```

## Client usage

The client only needs a meta model (no GPU required). All computation happens on the server.

### Blocking (default)

Same UX as local nnsight — `.save()` values are available immediately after the `with` block:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-30B-A3B")  # meta model, no GPU

with model.trace("The Eiffel Tower is in", serve="http://localhost:6677"):
    hidden = model.model.layers[24].output[0].save()
    logits = model.logits.output.save()

print(model.tokenizer.decode(logits.argmax(dim=-1)))  # Paris
print(hidden.shape)  # torch.Size([...])
```

### Non-blocking (concurrent)

Fire multiple requests concurrently. vLLM batches them for higher throughput:

```python
with model.trace("prompt 1", serve=url, blocking=False) as t1:
    out1 = model.logits.output.save()
with model.trace("prompt 2", serve=url, blocking=False) as t2:
    out2 = model.logits.output.save()

# Both in-flight concurrently inside vLLM's engine.
saves1 = t1.collect()  # blocks until response, returns {"out1": tensor}
saves2 = t2.collect()
```

Non-blocking mode returns saves as a dict from `.collect()` — variables are NOT auto-injected into the caller's scope.

### Multi-invoke (batched)

Multiple prompts in a single trace:

```python
with model.trace(temperature=0.0, top_p=1, serve=url) as tracer:
    with tracer.invoke("The capital of France is"):
        logits_fr = model.logits.output.save()
    with tracer.invoke("The capital of Japan is"):
        logits_jp = model.logits.output.save()
```

### Interventions

```python
with model.trace("The Eiffel Tower is in", serve=url):
    model.model.layers[40].output[0][:] = 0  # zero out layer 40
    logits = model.logits.output.save()
```

### With API key

```python
with model.trace("Hello", serve=url, api_key="mysecret"):
    logits = model.logits.output.save()
```

### Saving lists of activations

When saving multiple values into a collection, `.save()` the collection — not the individual elements:

```python
# CORRECT
with model.trace("Hello", serve=url):
    results = list().save()
    for i in range(12):
        results.append(model.model.layers[i].output[0])

# CORRECT
with model.trace("Hello", serve=url):
    results = [model.model.layers[i].output[0] for i in range(12)].save()

# WRONG — elements are saved but the list is not; values will be lost
with model.trace("Hello", serve=url):
    results = [model.model.layers[i].output[0].save() for i in range(12)]
```

## Limitations

- **vLLM only** — no HuggingFace Transformers backend (yet).
- **Cross-trace tensor references** — passing a saved tensor from one serve trace into another is not supported. The tensor exists on the client but the server has no access to it. This produces a clear error; the server engine is not affected.
- **Non-blocking frame injection** — `blocking=False` cannot inject values into the caller's frame (the frame has moved on). Use `.collect()` to retrieve saves as a dict.
