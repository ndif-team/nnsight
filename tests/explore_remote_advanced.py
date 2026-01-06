#!/usr/bin/env python
"""Advanced remote execution exploration."""

import torch
import time
from nnsight import LanguageModel, CONFIG
import nnsight

# Set API key
CONFIG.API.APIKEY = "b5320d07763a43ad95644842a0d4ff09"

model = LanguageModel("openai-community/gpt2")

print("=" * 60)
print("ADVANCED REMOTE EXECUTION TESTS")
print("=" * 60)

print("\n" + "-" * 60)
print("Test 1: Non-blocking execution")
print("-" * 60)

# Submit without waiting
with model.trace("Hello world", remote=True, blocking=False) as tracer:
    output = model.lm_head.output.save()

# Get the backend
backend = tracer.backend
print(f"Job ID: {backend.job_id}")
print(f"Initial status: {backend.job_status}")

# Poll for result
for i in range(30):
    result = backend()
    if result is not None:
        print(f"Got result after {i+1} polls!")
        # Result contains the saved values - let's see what it looks like
        print(f"Result type: {type(result)}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        if isinstance(result, dict) and "output" in result:
            print(f"Output shape: {result['output'].shape}")
        break
    print(f"  Poll {i+1}: status = {backend.job_status}")
    time.sleep(0.5)
else:
    print("Job did not complete in time")

print("\n" + "-" * 60)
print("Test 2: Out of order error (remote)")
print("-" * 60)

try:
    with model.trace("Hello", remote=True):
        # Access layer 5 before layer 0 - should fail
        h5 = model.transformer.h[5].output[0].save()
        h0 = model.transformer.h[0].output[0].save()
    print("ERROR: Should have raised an exception!")
except Exception as e:
    print(f"Caught expected error: {type(e).__name__}")
    print(f"Message preview: {str(e)[:200]}...")

print("\n" + "-" * 60)
print("Test 3: Saving tensors with detach().cpu()")
print("-" * 60)

with model.trace("Hello", remote=True):
    # Best practice - detach and move to CPU
    hidden = model.transformer.h[0].output[0].detach().cpu().save()

print(f"Hidden device: {hidden.device}")
print(f"Hidden shape: {hidden.shape}")
print(f"Requires grad: {hidden.requires_grad}")

print("\n" + "-" * 60)
print("Test 4: Multiple invokes in single trace")
print("-" * 60)

with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        paris_logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

    with tracer.invoke("The Statue of Liberty is in"):
        ny_logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

# Wait this should be local... let me try remote
print("(Running locally since remote=False)")
print(f"Paris prediction: {model.tokenizer.decode(paris_logit)}")
print(f"NY prediction: {model.tokenizer.decode(ny_logit)}")

# Now try remote with invokes
print("\nNow with remote=True:")
with model.trace(remote=True) as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        paris_logit2 = model.lm_head.output[0][-1].argmax(dim=-1).save()

    with tracer.invoke("The Statue of Liberty is in"):
        ny_logit2 = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(f"Paris prediction (remote): {model.tokenizer.decode(paris_logit2)}")
print(f"NY prediction (remote): {model.tokenizer.decode(ny_logit2)}")

print("\n" + "=" * 60)
print("ALL ADVANCED TESTS COMPLETED")
print("=" * 60)
