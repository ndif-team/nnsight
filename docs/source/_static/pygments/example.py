import json
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm

aligned = [
    "truth_teller",
    "genie",
    "saint",
]

misaligned = [
    "reward_maximizer",
    "fitness_maximizer",
    "money_maximizer",
]
