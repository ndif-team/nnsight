import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from baukit import nethook
import warnings

def untuple(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def print_formatted_results(prompts, txt, ret_dict):
    for i in range(len(prompts)):
        print(prompts[i])
        print(txt[i])
        if('answer' in ret_dict):
            answer = ret_dict['answer'][i]['candidates']
            print("p(answer): ", ", ".join([f"p(\'{t['token']}\'[{t['token_id']}])={t['p']}" for t in answer]))
        if('p_interesting_words' in ret_dict):
            p_interesting = ret_dict['p_interesting_words'][i]
            print("p(interesting): ", ", ".join([f"p(\'{t['token']}\'[{t['token_id']}])={t['p']}" for t in p_interesting]))

        print()


import unicodedata
from typing import Optional, List
import collections
import numpy as np



def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    top_k: int = 5,
    max_out_len: int = 20,
    max_new_tokens = None,
    argmax_greedy = False,
    debug = False,

    get_answer_tokens = False,      # returns the immediate next top token and `top_k` possible candidates
    track_interesting_words = None, # for each prompt tracks the p(token) of some interesting tokens as answer (the first generated token). 
                                    # `get_answer_tokens` must be true
    unoptimized_models = ["llama", "galactica"], # models that don't support `use_cache = True` and thus can't use fast generation
    use_cache = True
):
    # print(prompts)
    if(type(prompts) == str):
        prompts = [prompts]
        
    tokenized = tok(prompts, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )

    # print(tokenized['input_ids'].shape)
    input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    # print(cur_context)

    if get_answer_tokens == True:
        prompt_lens = np.array([
            tok([p], return_tensors="pt").input_ids.shape[-1]
            for p in prompts
        ])
        answers = [{'top_token': "<#>", 'candidates': []} for _ in range(input_ids.shape[0])]
        if(track_interesting_words is not None):
            p_interesting_words = [[] for _ in range(input_ids.shape[0])]

    if(max_new_tokens != None):
        max_out_len = input_ids.size(1) + max_new_tokens

    for unop in unoptimized_models:
        if(unop in model.config._name_or_path):
            use_cache = False
            warnings.warn(f"The model `{type(model)}` can't utilize `use_cache` for fast generation. Setting `use_cache = False`.")
            break

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache = use_cache,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            # print(" ====> ", logits.shape)

            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

            if(argmax_greedy == False):
                new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
                new_toks = torch.gather(tk, 1, new_tok_indices)

            else:
                new_tok_indices = torch.topk(softmax_out_top_k, dim=1, k=1)
                new_toks = torch.gather(tk, 1, new_tok_indices.indices)
            
            if(get_answer_tokens == True):
                for i in range(input_ids.shape[0]):
                    if(prompt_lens[i] == cur_context.stop):
                        answers[i]['top_token'] = tok.decode(new_toks[i][0])
                        for t in tk[i]:
                            answers[i]['candidates'].append(
                                {'token': tok.decode(t), 'token_id': t.item(), 'p': round(float(softmax_out[i][int(t)]), 4)}
                            )
                        if(track_interesting_words is not None):
                            for token in track_interesting_words[i]:
                                token_id = tok(token).input_ids[0]
                                p_interesting_words[i].append(
                                    {'token': tok.decode(token_id), 'token_id': token_id, 'p': round(float(softmax_out[i][token_id]), 4)}
                                )
                # print(answers)


            if(debug == True):
                for i in range(input_ids.size(0)):
                    # print(f"{i} => ", end="")
                    token_id = new_toks[i][0]
                    print(f"\'{tok.decode([token_id])}\'[{token_id}] -- {softmax_out[i][token_id]*100}", end=" ")
                    print("[", end="")
                    for t in tk[i]:
                        # print(t)
                        print(f"\'{tok.decode(t)}\'({round(float(softmax_out[i][int(t)]*100), 3)})", end=" ")
                    print("]")

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            if(use_cache == False):
                cur_context = slice(0, cur_context.stop + 1)
            else:
                cur_context = slice(cur_context.stop, cur_context.stop + 1)


    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n", " ")
        # .replace("<|endoftext|>", "")
        for x in txt
    ]

    # print(answers)

    # ret_dict = {"past_key_values": past_key_values}
    ret_dict = {}
    if(get_answer_tokens == True):
        ret_dict['answer'] = answers
        if(track_interesting_words is not None):
            ret_dict['p_interesting_words'] = p_interesting_words
    return txt, ret_dict
