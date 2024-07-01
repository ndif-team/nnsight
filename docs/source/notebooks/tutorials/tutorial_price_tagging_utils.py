import itertools
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from typing import Dict, Optional, Sequence
from torch.nn import functional as F
import re
import os, random, argparse, sys, pickle, time, datasets, json
import copy, torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from collections import Counter
import networkx as nx
import ipywidgets as widgets
from ipywidgets import interact
from matplotlib.patches import Rectangle

IGNORE_INDEX = -100

"""
This is for tutorial

If the cost is between X and Y.ipynb

These dataset creation functions are copied from
https://github.com/frankaging/pyvene/blob/cf93a1a6491dba65e1422fe20428f5972d17137e/counterfactual_datasets/price_tagging_game.py
"""

alpaca_prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""


def pricing_tag_game_config_sampler(amount, lower_bound, bound_width):
    if bound_width == None:
        bound_width_sample = round(random.uniform(2.50, 7.50), 2)
    else:
        bound_width_sample = bound_width
    if lower_bound == None:
        lower_bound_sample = round(random.uniform(0.05, 9.95 - bound_width_sample), 2)
        # left a little room to cover corner cases.
    else:
        lower_bound_sample = lower_bound
    upper_bound_sample = bound_width_sample + lower_bound_sample
    if amount == None:
        amount_sample = round(random.uniform(0.01, 9.99), 2)
    else:
        amount_sample = amount

    return lower_bound_sample, upper_bound_sample, amount_sample


def pricing_tag_game_example_sampler(
    tokenizer,
    amount,
    lower_bound,
    bound_width,
):
    (
        lower_bound_sample,
        upper_bound_sample,
        amount_sample,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)
    lower_bound_str = "%.2f" % lower_bound_sample
    upper_bound_str = "%.2f" % upper_bound_sample
    if amount_sample >= float(lower_bound_str) and amount_sample <= float(
        upper_bound_str
    ):
        label = tokenizer.convert_tokens_to_ids("Yes")
    else:
        label = tokenizer.convert_tokens_to_ids("No")

    amount_str = "%.2f dollars" % amount_sample
    instruction = f"Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no."
    alpaca_prompt = alpaca_prompt_template % (instruction, amount_str)
    input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0]) * -100).long().tolist()
    output_ids[-1] = label
    input_ids = input_ids.tolist()
    assert len(input_ids) == 82

    return input_ids, output_ids


def pricing_tag_game_example_sampler_with_info(
    tokenizer,
    amount,
    lower_bound,
    bound_width,
):
    (
        lower_bound_sample,
        upper_bound_sample,
        amount_sample,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)
    lower_bound_str = "%.2f" % lower_bound_sample
    upper_bound_str = "%.2f" % upper_bound_sample
    if amount_sample >= float(lower_bound_str) and amount_sample <= float(
        upper_bound_str
    ):
        label = tokenizer.convert_tokens_to_ids("Yes")
    else:
        label = tokenizer.convert_tokens_to_ids("No")

    amount_str = "%.2f dollars" % amount_sample
    instruction = f"Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no."
    alpaca_prompt = alpaca_prompt_template % (instruction, amount_str)
    input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0]) * -100).long().tolist()
    output_ids[-1] = label
    input_ids = input_ids.tolist()
    assert len(input_ids) == 82

    return (
        input_ids,
        output_ids,
        (lower_bound_sample, upper_bound_sample, amount_sample),
    )


def factual_sampler(
    tokenizer,
    max_n_training_examples,
    game="pricing_tag",
    amount=None,
    lower_bound=None,
    bound_width=None,
):
    all_input_ids = []
    all_output_ids = []  # this one does not have input ids, etc..
    for _ in range(max_n_training_examples):
        if "pricing_tag" in game:
            input_ids, output_ids = pricing_tag_game_example_sampler(
                tokenizer, amount, lower_bound, bound_width
            )
        elif game == "continent_retrieval":
            pass
        all_input_ids += [input_ids]
        all_output_ids += [output_ids]

    return all_input_ids, all_output_ids


def sample_with_region(region, lower_bound_sample, upper_bound_sample):
    if region == 1:
        amount_sample = round(random.uniform(0.01, lower_bound_sample - 0.01), 2)
    elif region == 2:
        amount_sample = round(random.uniform(lower_bound_sample, upper_bound_sample), 2)
    elif region == 3:
        amount_sample = round(random.uniform(upper_bound_sample + 0.01, 9.99), 2)
    return amount_sample


def lower_bound_alignment_example_sampler(
    tokenizer, amount=None, lower_bound=None, bound_width=None
):
    (
        base_lower_bound_sample,
        base_upper_bound_sample,
        _,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)
    (
        source_lower_bound_sample,
        source_upper_bound_sample,
        _,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)

    ctf_label_str = random.choice(["Yes", "No"])
    if ctf_label_str == "Yes":
        ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        base_source_regions = [
            [1, 2],
            [1, 3],
            [2, 2],
            [2, 3],
        ]
    elif ctf_label_str == "No":
        ctf_label = tokenizer.convert_tokens_to_ids("No")
        base_source_regions = [[1, 1], [2, 1], [3, 1], [3, 2], [3, 3]]
    base_source_region = random.choice(base_source_regions)
    base_region = base_source_region[0]
    source_region = base_source_region[1]

    base_amount_sample = sample_with_region(
        base_region, base_lower_bound_sample, base_upper_bound_sample
    )
    source_amount_sample = sample_with_region(
        source_region, source_lower_bound_sample, source_upper_bound_sample
    )

    return (
        base_lower_bound_sample,
        base_upper_bound_sample,
        source_lower_bound_sample,
        source_upper_bound_sample,
        base_amount_sample,
        source_amount_sample,
        ctf_label,
        ctf_label_str,
    )


def bound_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    bound_functors,
    amount=None,
    lower_bound=None,
    bound_width=None,
):
    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = []  # this one does not have input ids, etc..
    all_intervention_ids = []

    for _ in range(max_n_training_examples):
        bound_functor = random.choice(bound_functors)
        (
            base_lower_bound_sample,
            base_upper_bound_sample,
            source_lower_bound_sample,
            source_upper_bound_sample,
            base_amount_sample,
            source_amount_sample,
            ctf_label,
            ctf_label_str,
        ) = bound_functor(
            tokenizer,
            amount,
            lower_bound,
            bound_width,
        )

        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample

        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")

        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."

        base_alpaca_prompt = alpaca_prompt_template % (
            base_instruction,
            base_amount_str,
        )
        source_alpaca_prompt = alpaca_prompt_template % (
            source_instruction,
            source_amount_str,
        )

        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(
            source_alpaca_prompt, return_tensors="pt"
        ).input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids)) * -100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        intervention_id = 0 if bound_functor == bound_functors[0] else 1

        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]

        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [intervention_id]

        assert len(base_input_ids) == 82
        assert len(source_input_ids) == 82

    return (
        all_base_input_ids,
        all_source_input_ids,
        all_ctf_output_ids,
        all_intervention_ids,
    )


def midpoint_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    amount=None,
    lower_bound=None,
    bound_width=None,
):

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    
    for _ in range(max_n_training_examples):
        
        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        ctf_label = None
        ctf_label_str = None
        source_mid_point = (source_lower_bound_sample+source_upper_bound_sample)/2.0
        base_half = 0.5*abs(base_upper_bound_sample-base_lower_bound_sample)
        ctf_mid_diff = abs(base_amount_sample-source_mid_point)
        if ctf_mid_diff <= base_half:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            ctf_label_str = "Yes"
        else:
            ctf_label = tokenizer.convert_tokens_to_ids("No")
            ctf_label_str = "No"
            
        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample
        
        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")
        
        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."
        
        base_alpaca_prompt = alpaca_prompt_template % (base_instruction, base_amount_str)
        source_alpaca_prompt = alpaca_prompt_template % (source_instruction, source_amount_str)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids))*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [0]
        assert len(base_input_ids) == 82
        assert len(source_input_ids) == 82
        
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids


def bracket_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    amount=None,
    lower_bound=None,
    bound_width=None,
):

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    
    for _ in range(max_n_training_examples):
        
        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        ctf_label = None
        ctf_label_str = None
        if base_amount_sample <= source_upper_bound_sample and base_amount_sample >= source_lower_bound_sample:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            ctf_label_str = "Yes"
        else:
            ctf_label = tokenizer.convert_tokens_to_ids("No")
            ctf_label_str = "No"
            
        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample
        
        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")
        
        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."
        
        base_alpaca_prompt = alpaca_prompt_template % (base_instruction, base_amount_str)
        source_alpaca_prompt = alpaca_prompt_template % (source_instruction, source_amount_str)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids))*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [0]
        assert len(base_input_ids) == 82
        assert len(source_input_ids) == 82
        
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids
