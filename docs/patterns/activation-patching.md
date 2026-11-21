---
title: Activation Patching
one_liner: Replace activations from one run (e.g. clean) into another (e.g. corrupt) at a specific module to measure that component's causal contribution.
tags: [pattern, interpretability, causal-mediation, patching]
related: [docs/usage/barrier.md, docs/usage/invoke-and-batching.md, docs/patterns/attribution-patching.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/barrier.py]
---

# Activation Patching

## What this is for

Activation patching (causal mediation analysis, "denoising") asks: **does this
component carry the information that determines the answer?** Run a **clean** prompt
that produces the correct answer and a **corrupt** prompt that produces a different
one, take an activation from the clean run and paste it into the corrupt run at the
same module and position. If the corrupt run now predicts the clean answer, that
activation was sufficient to flip the model. Swapping the roles (corrupt into clean,
"noising") is the same code with the prompts exchanged.

In nnsight this is two `tracer.invoke(...)` calls in one `trace()`, with a
`tracer.barrier(n)` to hand the value across — see [barrier](../usage/barrier.md).
On `VLLM`, patching takes two traces instead — see
[Passing values between invokes](../models/vllm.md#passing-values-between-invokes).

Use it to localize where a fact or behavior lives (layer, position, head), to
confirm that a candidate from a probe or attention pattern is causal, and for IOI or
factual-recall designs with paired prompts.

## Canonical pattern

Patch the residual stream at the **subject-token positions** from a clean prompt
into a corrupt one. (Equal-length prompts let you patch matching positions.)

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

clean   = "The Eiffel Tower is in the city of"   # next token: " Paris"
corrupt = "The Colosseum is in the city of"      # both prompts are 10 tokens

paris = model.tokenizer.encode(" Paris")[0]
rome  = model.tokenizer.encode(" Rome")[0]

LAYER = 0
SUBJECT = slice(1, 5)          # the subject tokens differ between the two prompts

with model.trace() as tracer:
    barrier = tracer.barrier(2)

    # Invoke 1: capture the clean residual stream at LAYER.
    with tracer.invoke(clean):
        clean_hs = model.transformer.h[LAYER].output
        barrier()                       # signal: clean_hs is ready

    # Invoke 2: corrupt run, with the clean subject activations patched in.
    with tracer.invoke(corrupt):
        barrier()                       # wait until clean_hs is materialized
        model.transformer.h[LAYER].output[:, SUBJECT, :] = clean_hs[:, SUBJECT, :]
        patched_logits = model.lm_head.output[:, -1, :].save()

    # Invoke 3: corrupt-only baseline, no patching.
    with tracer.invoke(corrupt):
        baseline_logits = model.lm_head.output[:, -1, :].save()

pb, pp = baseline_logits.softmax(-1), patched_logits.softmax(-1)
print(f"baseline corrupt: P(Paris)={pb[0,paris]:.3f} P(Rome)={pb[0,rome]:.3f}")
print(f"patched  corrupt: P(Paris)={pp[0,paris]:.3f} P(Rome)={pp[0,rome]:.3f}")
```

```
baseline corrupt: P(Paris)=0.003 P(Rome)=0.014
patched  corrupt: P(Paris)=0.064 P(Rome)=0.006
```

GPT-2 small is near-degenerate on the corrupt prompt (`' Rome'` ranks only 4th), so
compare probabilities or the logit difference, not the top token.

!!! warning "Do not score with raw logits"

    GPT-2's logits carry a large shared offset that moves between prompts. On this
    pair the clean `' Paris'` logit (`-92.09`) is *below* the corrupt one (`-91.96`),
    so a raw-logit "recovery" has the wrong sign. Use probability, log-prob, or the
    `Paris - Rome` logit difference (`+2.42` clean, `-1.49` corrupt), normalized as
    `(patched - corrupt) / (clean - corrupt)`.

The value handed across by the barrier needs no `.clone()`: the write lands
correctly without it. Clone only when you will also modify the source tensor.

## Variations

### Find named tokens and align shifted prompts

For a fast tokenizer, use character offsets from the same encoding you pass to
the model. A substring can cross token boundaries or start inside a token.
Select every token that overlaps the character span, excluding special tokens
and empty offsets. Occurrences are counted from zero, from left to right,
without overlap.

```python
def positions_for(tokenizer, prompt, substring, occurrence=0):
    if not tokenizer.is_fast:
        raise ValueError("positions_for requires a fast tokenizer with offsets")
    if not substring or occurrence < 0:
        raise ValueError("provide a nonempty substring and a nonnegative occurrence")

    end = 0
    for _ in range(occurrence + 1):
        start = prompt.find(substring, end)
        if start == -1:
            raise ValueError(f"occurrence {occurrence} of {substring!r} is missing")
        end = start + len(substring)

    encoding = tokenizer(
        prompt, return_offsets_mapping=True, return_special_tokens_mask=True
    )
    positions = [
        index
        for index, ((left, right), special) in enumerate(
            zip(encoding.offset_mapping, encoding.special_tokens_mask, strict=True)
        )
        if not special and left < right and left < end and start < right
    ]
    if not positions:
        raise ValueError(f"no tokens overlap occurrence {occurrence} of {substring!r}")
    return encoding.input_ids, positions


def align_prompt_ids(clean_ids, corrupt_ids):
    """Return (clean, corrupt) pairs and the two skipped half-open ranges."""
    prefix = 0
    for clean_id, corrupt_id in zip(clean_ids, corrupt_ids):
        if clean_id != corrupt_id:
            break
        prefix += 1

    clean_end, corrupt_end = len(clean_ids), len(corrupt_ids)
    while (
        clean_end > prefix
        and corrupt_end > prefix
        and clean_ids[clean_end - 1] == corrupt_ids[corrupt_end - 1]
    ):
        clean_end -= 1
        corrupt_end -= 1

    pairs = [(index, index) for index in range(prefix)]
    pairs.extend(
        zip(range(clean_end, len(clean_ids)), range(corrupt_end, len(corrupt_ids)))
    )
    return pairs, (prefix, clean_end), (prefix, corrupt_end)
```

Prefix and suffix alignment pairs unchanged token IDs after a replacement,
insertion, or deletion. The skipped ranges describe the remaining middle,
including any shared tokens inside it. To patch the changed subjects
themselves, use their separately resolved positions and an explicit pairing.

This example swaps one aligned token in the suffix after a changed subject.
The traces each take one unpadded sequence, so the token indices also index
the activation tensors. In a padded batch, use each row's attention mask to
translate prompt-relative indices to tensor positions.

```python
from nnsight import save

clean = "The capital of Germany is"
corrupt = "The capital of the United States is"
clean_ids, clean_subject = positions_for(model.tokenizer, clean, "Germany")
corrupt_ids, corrupt_subject = positions_for(
    model.tokenizer, corrupt, "the United States"
)
pairs, clean_skipped, corrupt_skipped = align_prompt_ids(clean_ids, corrupt_ids)
print("skipped token ranges", clean_skipped, corrupt_skipped)

suffix_pairs = [
    (source, dest)
    for source, dest in pairs
    if source > clean_subject[-1] and dest > corrupt_subject[-1]
]
if not suffix_pairs:
    raise ValueError("the prompts have no aligned tokens after the subjects")
source, dest = suffix_pairs[0]
with model.trace(clean_ids):
    donor = save(model.transformer.h[LAYER].output[:, source, :].detach())
with model.trace(corrupt_ids):
    model.transformer.h[LAYER].output[:, dest, :] = donor
    patched_logits = save(model.lm_head.output[:, -1, :])
```

### Sweep over layers

```python
results = {}
for layer in range(len(model.transformer.h)):
    with model.trace() as tracer:
        barrier = tracer.barrier(2)
        with tracer.invoke(clean):
            hs_clean = model.transformer.h[layer].output
            barrier()
        with tracer.invoke(corrupt):
            barrier()
            model.transformer.h[layer].output[:, SUBJECT, :] = hs_clean[:, SUBJECT, :]
            logits = model.lm_head.output[:, -1, :].save()
    results[layer] = logits.softmax(-1)[0, paris].item()
```

`P(Paris)` decays monotonically from `0.064` at layer 0 to `0.003` at layer 11.

The same sweep fits in **one** trace: one clean invoke, one corrupt invoke per layer
and a baseline, with one barrier round per layer (see
[barrier](../usage/barrier.md#several-sites-one-round-each-and-fence-every-read)).
It matches the per-layer traces to every printed digit:

```python
import nnsight

N = len(model.transformer.h)
with model.trace() as tracer:
    barrier = tracer.barrier(N + 1)              # the clean invoke + one per layer
    results = nnsight.save({})
    with tracer.invoke(clean):
        hs = []
        for block in model.transformer.h:
            hs.append(block.output)
            barrier()                            # round L: layer L is ready
    for layer in range(N):
        with tracer.invoke(corrupt):
            for r in range(N):                   # attend every round, write in its own
                barrier()
                if r == layer:
                    model.transformer.h[layer].output[:, SUBJECT, :] = hs[layer][:, SUBJECT, :]
            results[layer] = model.lm_head.output[0, -1].softmax(-1)[paris].item()
    with tracer.invoke(corrupt):                 # baseline: no barrier
        results["baseline"] = model.lm_head.output[0, -1].softmax(-1)[paris].item()
```

!!! warning "A whole-span residual sweep measures reach, not location"

    Patching `block.output` at layer L overwrites everything computed at those
    positions up to L. Patch the embeddings and you reproduce the clean run; patch the
    last block and a non-final position has no path left to the logits. A monotone
    decay between those ends is the null result, not a finding — patching the
    identical *suffix* tokens gives the mirror-image curve. Use the layer × position
    map below to localize.

### Layer × position map

Patching one (layer, position) cell at a time separates what the sweep conflates.
Cache the clean run once and batch a whole layer's positions into one forward pass:

```python
import nnsight

with model.trace(clean):
    clean_resid = nnsight.save([block.output.detach() for block in model.transformer.h])

corrupt_ids = model.tokenizer(corrupt).input_ids
tokens = [model.tokenizer.decode([i]) for i in corrupt_ids]

grid = []
for layer in range(0, 12, 2):
    with model.trace() as tracer:
        row = nnsight.save([])
        for pos in range(len(corrupt_ids)):
            with tracer.invoke(corrupt):
                model.transformer.h[layer].output[:, pos, :] = clean_resid[layer][:, pos, :]
                # .detach(): drop the graph so the saved values don't retain it
                row.append(model.output.logits[0, -1].softmax(-1)[paris].detach())
    grid.append([float(x) for x in row])

print(f"{'position':<12}" + "".join(f"L{l:<6}" for l in range(0, 12, 2)))
for pos, token in enumerate(tokens):
    print(f"{token!r:<12}" + "".join(f"{grid[i][pos]:.3f}  " for i in range(len(grid))))
```

```
position    L0     L2     L4     L6     L8     L10
'The'       0.003  0.003  0.003  0.003  0.003  0.003
' Col'      0.002  0.003  0.004  0.003  0.003  0.003
'os'        0.007  0.020  0.011  0.005  0.005  0.003
'se'        0.002  0.002  0.005  0.006  0.005  0.004
'um'        0.006  0.007  0.008  0.007  0.005  0.003
' is'       0.003  0.003  0.003  0.003  0.003  0.003
' in'       0.003  0.003  0.003  0.004  0.003  0.003
' the'      0.003  0.003  0.003  0.003  0.003  0.003
' city'     0.004  0.005  0.004  0.005  0.003  0.003
' of'       0.003  0.003  0.004  0.006  0.017  0.052
```

Against the corrupt baseline of `0.003`, two sites separate out: the subject pieces
`'os'` and `'um'` matter at layers 0–4 and stop mattering by layer 8, and the final
`' of'` position does nothing until layer 8 and then carries the answer. The suffix
positions are flat everywhere, which is what the sweep alone could not show.

### Patch the last position instead

Patch the final position instead of the subject to ask "is the *final* prediction
sensitive to this layer?" — the corrupt invoke of the per-layer sweep becomes:

```python
barrier()
model.transformer.h[layer].output[:, -1, :] = hs_clean[:, -1, :]
logits = model.lm_head.output[:, -1, :].save()
```

On this pair `P(Paris)` stays near the corrupt `0.003` through layer 7 (`0.007`),
then rises to the full clean `0.070` by layer 11. Patching the subject localizes the *source* of a fact; patching the last
position tests the *sensitivity* of the final prediction.

### Per-head or sub-block patching

Patch `attn.output[0]` or `mlp.output` instead of the block residual to isolate one
sub-block; for a single head, patch a slice of `attn.c_proj.input` — see
[per-head-attention](per-head-attention.md).

## Interpretation tips

- **Always run a no-patch baseline** for the corrupt prompt in the same `trace()`;
  it is the reference.
- **Effects can be small.** A few-percent shift on a sharp prompt is often real.
  Average over several prompt pairs.
- **Barrier count = number of invokes that call `barrier()`.** The no-patch
  baseline invoke does not synchronize and is not counted.

## Gotchas

- Clean and corrupt prompts of **different token lengths** can't share an absolute
  position slice, and padding shifts absolute indices with whatever else is in the
  batch; `[:, -1, :]` survives left padding. See
  [invoke-and-batching](../usage/invoke-and-batching.md).
- Barrier semantics, forward-order access, tuple outputs, and cloning before an
  in-place write: [barrier](../usage/barrier.md),
  [out-of-order-error](../errors/out-of-order-error.md),
  [access-and-modify](../usage/access-and-modify.md).

## Related

- [attribution-patching](attribution-patching.md) — linear approximation that gets a
  whole causal map from one clean + one corrupt run.
- [invoke-and-batching](../usage/invoke-and-batching.md#comparing-prompts-one-trace-not-a-loop-of-traces)
- https://nnsight.net/notebooks/tutorials/activation_patching/
- Meng et al. (2022), "Locating and Editing Factual Associations in GPT" (ROME).
- Wang et al. (2022), "Interpretability in the Wild" (IOI).
