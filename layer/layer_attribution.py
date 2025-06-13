# Standard imports
import os
from evaluate_agent import evaluate_agent
import plotly.express as px
from transformer_lens.hook_points import HookPoint
from IPython.display import IFrame
import json
import requests
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import Information_Entropy
import torch
from torch import Tensor
from accelerate import Accelerator
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
import shap
from typing import List
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Bool, Float, Int
from rich import print as rprint
from rich.table import Column, Table


from datasets import load_dataset
import re
import time
from transformers.utils import logging
logging.set_verbosity_error()
import logging
from functools import partial
logging.getLogger('shap').setLevel(logging.ERROR)

torch.set_grad_enabled(False)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer


torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)

# Here is where we test on a single prompt
# Result: 70% probability on Mary, as we expect

# example_prompt = "vector vector vector vector vector vector vector vector vector vector vector vector vector vector vector vector vector vector"
# example_answer = " vector"
example_prompt = "School is a place where you school is a school is a school is a" # 成了
example_answer = " school"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

prompts = [
    "Is displacement is a vector or scalar is a vector is is a vector is is",
    "School school school is a place where you school school school is a school is a school is a",
    "Which does not has an index does not has an index",
    "Friends friends friends are people who friends friends friends are people who friends are people who",
    "Speed speed speed is a scalar that speed speed speed is a speed is a speed is a",
    "Mass mass mass does not change mass mass mass changes doesn't change",
    "Work done is energy is energy is",
    "Time is always measured in seconds Time is always measured in seconds Time is always",
]
answers = [
    [" a", " vector"],
    [" school", " place"],
    [" does", " and"],
    [" friends", " help"],
    [" speed", " scalar"],
    [" mass", " anything"],
    [" energy", " to"],
    [" measured", " limited"]
]

answer_tokens = torch.concat([
    model.to_tokens(words, prepend_bos=False).T for words in answers
])
expected_shape = len(answers)
rprint(prompts)
rprint(answers)
answer_tokens = answer_tokens[~(answer_tokens == 50256).any(dim=1)]
rprint(answer_tokens)
table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")

for prompt, answer in zip(prompts, answers):
    table.add_row(prompt, repr(answer[0]), repr(answer[1]))

rprint(table)

tokens = model.to_tokens(prompts, prepend_bos=True)
# Move the tokens to the GPU
tokens = tokens.to(device)
# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens) # [batch 2 d_model]
print("Answer residual directions shape:", answer_residual_directions.shape)

correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
logit_diff_directions = correct_residual_directions - incorrect_residual_directions # [batch d_model]
print(f"Logit difference directions shape:", logit_diff_directions.shape)

def residual_stack_to_logit_diff(

        residual_stack: Float[Tensor, "... batch d_model"],
        cache: ActivationCache,
        logit_diff_directions: Float[Tensor, "batch d_model"]
) -> Float[Tensor, "..."]:

    # calculate every layer's logit difference contribution
    layer_contributions = []
    for layer_residual in residual_stack:
       
        logit_contribution = torch.sum(layer_residual * logit_diff_directions, dim=-1)
        layer_contributions.append(logit_contribution)

    # concatenate each layer's contribution to get a [component, batch] tensor
    logit_diff_per_layer = torch.stack(layer_contributions, dim=0)

    return logit_diff_per_layer



per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
print(labels)
print(per_layer_residual.shape)
print(logit_diff_directions.shape)
# calculate the logit difference
logit_lens_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache, logit_diff_directions)

print(f"Logit differences per layer: {logit_lens_logit_diffs}")
