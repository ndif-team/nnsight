from nnsight import LanguageModel
import torch
import einops
import circuitsvis as cv

model = LanguageModel("gpt2", device_map='cuda:0', dispatch=True)

prompts = ['When John and Mary went to the shops, John gave the bag to', 'When John and Mary went to the shops, Mary gave the bag to', 'When Tom and James went to the park, James gave the ball to', 'When Tom and James went to the park, Tom gave the ball to', 'When Dan and Sid went to the shops, Sid gave an apple to', 'When Dan and Sid went to the shops, Dan gave an apple to', 'After Martin and Amy went to the park, Amy gave a drink to', 'After Martin and Amy went to the park, Martin gave a drink to']
answers = [(' Mary', ' John'), (' John', ' Mary'), (' Tom', ' James'), (' James', ' Tom'), (' Dan', ' Sid'), (' Sid', ' Dan'), (' Martin', ' Amy'), (' Amy', ' Martin')]

clean_tokens = model.tokenizer(prompts, return_tensors="pt")['input_ids']
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i+1 if i%2==0 else i-1) for i in range(len(clean_tokens)) ]
    ]

answer_token_indices = torch.tensor([[model.tokenizer(answers[i][j])['input_ids'][0] for j in range(2)] for i in range(len(answers))], device=model.local_model.device)

def ioi_metric(logits,  CLEAN_BASELINE, CORRUPTED_BASELINE, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape)==3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

with model.forward(inference=False) as runner:
    with runner.invoke(clean_tokens) as invoker:
        
        clean_logits = model.lm_head.output

        clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item().save()

        clean_cache = [model.transformer.h[i].attn.attn_dropout.input.save() for i in range(len(model.transformer.h))]

        clean_grad_cache = [model.transformer.h[i].attn.attn_dropout.backward_input.save() for i in range(len(model.transformer.h))]

    with runner.invoke(corrupted_tokens) as invoker:
        
        corrupted_logits = model.lm_head.output

        corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item().save()

        corrupted_cache = [model.transformer.h[i].attn.attn_dropout.input.save() for i in range(len(model.transformer.h))]
 
        clean_value = ioi_metric(clean_logits, clean_logit_diff, corrupted_logit_diff).save()

        clean_value.backward()

# Ideally could do that in one forward. Cant because two backward calls would hit the same proxy
# Need to rethink invoker.next() and maybe add module.next()
with model.forward(inference=False) as runner:
    with runner.invoke(corrupted_tokens) as invoker:

        corrupted_logits = model.lm_head.output
        
        corrupted_grad_cache = [model.transformer.h[i].attn.attn_dropout.backward_input.save() for i in range(len(model.transformer.h))]

        corrupted_value = ioi_metric(corrupted_logits, clean_logit_diff.value, corrupted_logit_diff.value).save()

        corrupted_value.backward()

clean_cache = torch.stack([value.value[0] for value in clean_cache])
clean_grad_cache = torch.stack([value.value[0] for value in clean_grad_cache])
corrupted_cache = torch.stack([value.value[0] for value in corrupted_cache])
corrupted_grad_cache = torch.stack([value.value[0] for value in corrupted_grad_cache])

print("Clean Value:", clean_value.value.item())
print("Corrupted Value:", corrupted_value.value.item())

def create_attention_attr(clean_cache, clean_grad_cache):
    attention_attr = clean_grad_cache * clean_cache
    attention_attr = einops.rearrange(attention_attr, "layer batch head_index dest src -> batch layer head_index dest src")
    return attention_attr

attention_attr = create_attention_attr(clean_cache, clean_grad_cache)

n_layers = len(model.transformer.h)
n_heads = model.transformer.h[0].attn.num_heads

HEAD_NAMES = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
HEAD_NAMES_QKV = [f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]]
print(HEAD_NAMES[:5])
print(HEAD_NAMES_SIGNED[:5])
print(HEAD_NAMES_QKV[:5])

def plot_attention_attr(attention_attr, tokens, top_k=20, index=0, title=""):
    if len(tokens.shape)==2:
        tokens = tokens[index]
    if len(attention_attr.shape)==5:
        attention_attr = attention_attr[index]
    attention_attr_pos = attention_attr.clamp(min=-1e-5)
    attention_attr_neg =  - attention_attr.clamp(max=1e-5)
    attention_attr_signed = torch.stack([attention_attr_pos, attention_attr_neg], dim=0)
    attention_attr_signed = einops.rearrange(attention_attr_signed, "sign layer head_index dest src -> (layer head_index sign) dest src")
    attention_attr_signed = attention_attr_signed / attention_attr_signed.max()
    attention_attr_indices = attention_attr_signed.max(-1).values.max(-1).values.argsort(descending=True)
    # print(attention_attr_indices.shape)
    # print(attention_attr_indices)
    attention_attr_signed = attention_attr_signed[attention_attr_indices, :, :]
    head_labels = [HEAD_NAMES_SIGNED[i.item()] for i in attention_attr_indices]

    tokens = [model.tokenizer.decode(token) for token in tokens]

    return cv.circuitsvis.attention.attention_heads(tokens=tokens, attention=attention_attr_signed[:top_k], attention_head_names=head_labels[:top_k])

plot_attention_attr(attention_attr, clean_tokens, index=0, title="Attention Attribution for first sequence")


        