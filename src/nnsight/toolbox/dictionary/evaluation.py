"""
Utilities for evaluating dictionaries on a model and datset.
"""

import torch as t
from .training import sae_loss
import matplotlib.pyplot as plt


def loss_recovered(
        tokens, # a tokenized batch
        model, # an nnsight LanguageModel
        submodules, # submodules of model
        dictionaries, # dictionaries for submodules
        io='out', # can be 'in', 'out', or 'in_to_out'
        pct=False # return pct recovered; if False, return losses
):
    """
    How much of the model's loss is recovered by replacing the component output 
    with the reconstruction by the autoencoder?
    """
    
    # unmodified logits
    with model.invoke(tokens) as invoker:
        pass
    logits_original = invoker.output.logits
    
    # logits when replacing component output with reconstruction by autoencoder
    with model.invoke(tokens) as invoker:
        for submodule, dictionary in zip(submodules, dictionaries):
            if io == 'in':
                if type(submodule.input.shape) == tuple:
                    submodule.input[0][:] = dictionary(submodule.input[0])
                else:
                    submodule.input = dictionary(submodule.input)
            elif io == 'out':
                if type(submodule.output.shape) == tuple:
                    submodule.output[0][:] = dictionary(submodule.output[0])
                else:
                    submodule.output = dictionary(submodule.output)
            elif io == 'in_to_out':
                if type(submodule.input.shape) == tuple:
                    submodule.output[0][:] = dictionary(submodule.input[0])
                else:
                    submodule.output = dictionary(submodule.input)
            else:
                raise ValueError(f"invalid io: {io}")
        
    logits_reconstructed = invoker.output.logits
    
    # logits when zero ablating components
    with model.invoke(tokens) as invoker:
        for submodule in submodules:
            if io == 'in':
                if type(submodule.input.shape) == tuple:
                    submodule.input[0][:] = t.zeros_like(submodule.input[0])
                else:
                    submodule.input = t.zeros_like(submodule.input)
            elif io == 'out':
                if type(submodule.output.shape) == tuple:
                    submodule.output[0][:] = t.zeros_like(submodule.output[0])
                else:
                    submodule.output = t.zeros_like(submodule.output)
            elif io == 'in_to_out':
                if type(submodule.input.shape) == tuple:
                    submodule.output[0][:] = t.zeros_like(submodule.input[0])
                else:
                    submodule.output = t.zeros_like(submodule.input)
            else:
                raise ValueError(f"invalid io: {io}")

    logits_zero = invoker.output.logits
    
    losses = []
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)(
            logits[:,:-1,:].reshape(-1, logits.shape[-1]),
            tokens[:,1:].reshape(-1)
        ).item()
        losses.append(loss)
    
    if pct:
        return (losses[1] - losses[2]) / (losses[0] - losses[2])
    else:
        return tuple(losses)

def evaluate(
        model, # a nnsight LanguageModel
        submodule, # a submodule of model
        dictionary, # a dictionary
        activations, # an ActivationBuffer
        entropy=False, # whether to use entropy regularization
        hist_save_path=None, # path for saving histograms
        hist_title=None, # title for histograms
        io='out', # can be 'in', 'out', or 'in_to_out'
        device='cpu'
):
    with t.no_grad():

        out = {} # dict of results

        acts = next(activations).to(device)

        # compute reconstruction (L2) loss and sparsity loss
        mse_loss, sparsity_loss = sae_loss(acts, dictionary, sparsity_penalty=None, use_entropy=entropy, separate=True)
        out['mse_loss'] = mse_loss.item() ** 2
        out['sparsity_loss'] = sparsity_loss.item()

        # compute mean L0 norm and percentage of neurons alive
        features = dictionary.encode(acts)
        actives = (features != 0)
        out['l0'] = actives.float().sum(dim=-1).mean().item()
        alives = actives.any(dim=0)
        out['percent_alive'] = alives.float().mean().item()

        # compute histogram if needed
        if hist_save_path is not None:
            freqs = actives.float().mean(dim=0)
            plt.figure()
            plt.hist(freqs.cpu(), bins=t.logspace(-5, 0, 100))
            plt.xscale('log')
            plt.title(hist_title)
            plt.savefig(hist_save_path)
            plt.close()
        
        # compute loss recovered
        tokens = activations.tokenized_batch().input_ids
        tokens = tokens.to(device)
        loss_original, loss_reconstructed, loss_zero = \
            loss_recovered(tokens, model, [submodule], [dictionary], io=io, pct=False)
        out['loss_original'] = loss_original
        out['loss_reconstructed'] = loss_reconstructed
        out['loss_zero'] = loss_zero
        out['percent_recovered'] = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)

        return out

# %%
