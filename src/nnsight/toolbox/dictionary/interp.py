import zstandard as zstd
import json
import os
import io
import random
from tqdm import tqdm
from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.training import trainSAE
from collections import defaultdict
from circuitsvis.activations import text_neuron_activations
from circuitsvis.topk_tokens import topk_tokens
from datasets import load_dataset
from einops import rearrange
import torch as t

def list_decode(model, x):
    if isinstance(x, int):
        return model.tokenizer.decode(x)
    else:
        return [list_decode(model, y) for y in x]
    

def random_feature(model, submodule, autoencoder, buffer,
                   num_examples=10):
    inputs = buffer.tokenized_batch()
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(inputs['input_ids'], scan=False) as invoker:
            hidden_states = submodule.output.save()
    dictionary_activations = autoencoder.encode(hidden_states.value)
    num_features = dictionary_activations.shape[2]
    feat_idx = random.randint(0, num_features-1)
    
    flattened_acts = rearrange(dictionary_activations, 'b n d -> (b n) d')
    acts = dictionary_activations[:, :, feat_idx].cpu()
    flattened_acts = rearrange(acts, 'b l -> (b l)')
    top_indices = t.argsort(flattened_acts, dim=0, descending=True)[:num_examples]
    batch_indices = top_indices // acts.shape[1]
    token_indices = top_indices % acts.shape[1]

    tokens = [
        inputs['input_ids'][batch_idx, :token_idx+1].tolist() for batch_idx, token_idx in zip(batch_indices, token_indices)
    ]
    tokens = list_decode(model, tokens)
    activations = [
        acts[batch_idx, :token_id+1, None, None] for batch_idx, token_id in zip(batch_indices, token_indices)
    ]

    return (feat_idx, tokens, activations)

def feature_effect(
        model,
        submodule,
        dictionary,
        feature,
        input_tokens,
        add_residual=True, # whether to compensate for dictionary reconstruction error by adding residual
        k=10,
        largest=True,
):
    """
    Effect of ablating the feature on top k predictions for next token.
    """
    # clean run
    with model.invoke(input_tokens) as invoker:
        if dictionary is None:
            pass
        elif not add_residual: # run hidden state through autoencoder
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:] = dictionary(submodule.output[0])
            else:
                submodule.output = dictionary(submodule.output)
    clean_logits = invoker.output.logits[0, -1, :]
    clean_logprobs = t.nn.functional.log_softmax(clean_logits, dim=-1)

    # ablated run
    with model.invoke(input_tokens) as invoker:
        if type(submodule.output.shape) == tuple:
            x = submodule.output[0]
        else:
            x = submodule.output

        if dictionary is None:
            if type(submodule.output.shape) == tuple:
                submodule.output[0][0, -1, feature] = 0
            else:
                submodule.output[0, -1, feature] = 0
        else:
            f = dictionary.encode(x)   
            f[0, -1, feature] = 0
            if not add_residual:
                x = dictionary.decode(f)
            else:
                residual = dictionary(x) - x
                x = dictionary.decode(f) - residual
            
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:] = x
            else:
                submodule.output = x
    
    ablated_logits = invoker.output.logits[0, -1, :]
    ablated_logprobs = t.nn.functional.log_softmax(ablated_logits, dim=-1)
    diff = clean_logprobs - ablated_logprobs

    top_probs, top_tokens = t.topk(diff, k=k, largest=largest)
    return top_tokens, top_probs


def examine_dimension(model, submodule, buffer, dictionary=None,
                      dim_idx=None, k=30):
    def _list_decode(x):
        if isinstance(x, int):
            return model.tokenizer.decode(x)
        else:
            return [_list_decode(y) for y in x]
        
    # are we working with residuals?
    is_resid = False
    with model.invoke("dummy text") as invoker:
        if type(submodule.output.shape) == tuple:
            is_resid = True
    
    if dictionary is not None:
        dimensions = dictionary.encoder.out_features
    else:
        dimensions = submodule.output[0].shape[-1] if is_resid else submodule.output.shape[-1]
    
    inputs = buffer.tokenized_batch().to("cuda")
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(inputs['input_ids'], scan=False) as invoker:
            hidden_states = submodule.output.save()
    hidden_states = hidden_states.value[0] if is_resid else hidden_states.value
    if dictionary is not None:
        activations = dictionary.encode(hidden_states)
    else:
        activations = hidden_states

    flattened_acts = rearrange(activations, 'b n d -> (b n) d')
    freqs = (flattened_acts !=0).sum(dim=0) / flattened_acts.shape[0]

    k = k
    if dim_idx is not None:
        feat = dim_idx
    else:
        feat = random.randint(0, dimensions-1)
    acts = activations[:, :, feat].cpu()
    flattened_acts = rearrange(acts, 'b l -> (b l)')
    topk_indices = t.argsort(flattened_acts, dim=0, descending=True)[:k]
    batch_indices = topk_indices // acts.shape[1]
    token_indices = topk_indices % acts.shape[1]
    
    tokens = [
        inputs['input_ids'][batch_idx, :token_idx+1].tolist() for batch_idx, token_idx in zip(batch_indices, token_indices)
    ]
    tokens = _list_decode(tokens)
    activations = [
        acts[batch_idx, :token_id+1, None, None] for batch_idx, token_id in zip(batch_indices, token_indices)
    ]

    token_acts_sums = defaultdict(float)
    token_acts_counts = defaultdict(int)
    token_mean_acts = defaultdict(float)
    for context_idx, context in enumerate(activations):
        for token_idx in range(activations[context_idx].shape[0]):
            token = tokens[context_idx][token_idx]
            activation = activations[context_idx][token_idx].item()
            token_acts_sums[token] += activation
            token_acts_counts[token] += 1
    for token in token_acts_sums:
        token_mean_acts[token] = token_acts_sums[token] / token_acts_counts[token]
    token_mean_acts = {k: v for k, v in sorted(token_mean_acts.items(), key=lambda item: item[1], reverse=True)}
    top_tokens = []
    i = 0
    for token in token_mean_acts:
        top_tokens.append((token, token_mean_acts[token]))
        i += 1
        if i >= 10:
            break

    top_contexts = text_neuron_activations(tokens, activations)

    # this isn't working as expected, for some reason
    top_affected = []
    affected_tokens, prob_change = feature_effect(model, submodule, dictionary, dim_idx, inputs)
    for idx, tok_idx in enumerate(affected_tokens):
        token = model.tokenizer._convert_id_to_token(tok_idx)
        prob = prob_change[idx].item()
        top_affected.append((token, prob))

    return {"top_contexts": top_contexts,
            "top_tokens": top_tokens,
            "top_affected": top_affected}