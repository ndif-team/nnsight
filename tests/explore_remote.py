#!/usr/bin/env python
"""Quick exploration of remote execution."""

import torch
from nnsight import LanguageModel, CONFIG
import nnsight

# Set API key
CONFIG.API.APIKEY = "b5320d07763a43ad95644842a0d4ff09"

print("=" * 60)
print("BASIC REMOTE EXECUTION TEST")
print("=" * 60)

print("\nLoading model on meta device...")
model = LanguageModel("openai-community/gpt2")
print(f"Model device: {model.device}")

print("\n" + "-" * 60)
print("Test 1: Basic remote trace")
print("-" * 60)

with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(f"\nPredicted token ID: {logit}")
print(f"Decoded: {model.tokenizer.decode(logit)}")

print("\n" + "-" * 60)
print("Test 2: Print statements in remote trace (watch for LOG)")
print("-" * 60)

with model.trace("Hello world", remote=True):
    hidden = model.transformer.h[0].output[0]
    print(f"=== REMOTE LOG: Hidden shape: {hidden.shape} ===")
    print(f"=== REMOTE LOG: Hidden mean: {hidden.mean()} ===")
    output = model.lm_head.output.save()

print(f"\nOutput shape: {output.shape}")

print("\n" + "-" * 60)
print("Test 3: Session with variable sharing")
print("-" * 60)

with model.session(remote=True):
    with model.trace("The Eiffel Tower is in"):
        paris_hidden = model.transformer.h[5].output[0][:, -1, :]  # No save

    with model.trace("The Statue of Liberty is in"):
        # Use paris_hidden directly
        model.transformer.h[5].output[0][:, -1, :] = paris_hidden
        patched_logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

    with model.trace("The Statue of Liberty is in"):
        clean_logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(f"\nClean prediction: {model.tokenizer.decode(clean_logit)}")
print(f"Patched prediction: {model.tokenizer.decode(patched_logit)}")

print("\n" + "-" * 60)
print("Test 4: Generation with iter")
print("-" * 60)

with model.generate("Once upon a time", max_new_tokens=5, remote=True) as tracer:
    tokens = list().save()
    with tracer.iter[:]:
        tokens.append(model.lm_head.output[0][-1].argmax(dim=-1))

    output = tracer.result.save()

print(f"\nGenerated tokens: {tokens}")
print(f"Decoded tokens: {[model.tokenizer.decode(t) for t in tokens]}")
print(f"Full output: {model.tokenizer.decode(output[0])}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
