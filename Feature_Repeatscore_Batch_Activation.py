# Standard imports
import random
import os
from evaluate_agent import evaluate_agent
# 批量设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/.catch'
os.environ['HF_CACHE_DIR'] = '/root/autodl-tmp/.catch'
os.environ['HF_HOME'] = '/root/autodl-tmp/.catch/huggingface'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp/.catch/huggingface/hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TOKEN'] = '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'
import sys
# 将父文件夹的路径添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_lens.hook_points import HookPoint
from IPython.display import IFrame
import json
import requests
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import metrics.Information_Entropy
import torch
from accelerate import Accelerator
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
import shap
from typing import List
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from datasets import load_dataset
import re
import time
from transformers.utils import logging
import logging
from functools import partial
import metrics.n_gram
import metrics.Information_Entropy
from jaxtyping import Float, Int
from torch import Tensor
from rich.table import Table
from rich import print as rprint
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens import SAE, HookedSAETransformer
import argparse
from evaluate import load

torch.set_grad_enabled(False)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

feature_id = [] # 所有的feature的列表

df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)


torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

def compute_metrics(input_sentence, metric):
    if metric == 'rs':
        return metrics.n_gram.calculate_rep_n(input_sentence, 1)
    if metric == 'ie':
        return metrics.Information_Entropy.calculate_ngram_entropy(input_sentence, 1)

def access_dataset(dataset):
    input_list = []
    ds = load_dataset(dataset)
    if 'Academic' in dataset:
        # dataset="DisgustingOzil/Academic_dataset_ShortQA"
        for i in range(1): # 调整测试的数量
            response = ds['train']['response'][i]
            # 使用正则表达式匹配 <question> 标签之间的内容
            question = re.search(r'<question>(.*?)</question>', response)
            question = question.group(1).strip()  # 打印匹配到的 question 内容
            input_list.append(question)
        return input_list

    if 'natural' in dataset:
        # dataset = "sentence-transformers/natural-questions"
        for i in range(1):
            question = ds['train']['query'][i]
            input_list.append(question)
        return input_list
    if 'Diversity' in dataset:
        for i in range(10):
            question = ds['train']['question'][i]
            input_list.append(question)
        return input_list

def steering_hook(
    activations: Float[Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idxs: list[int],  # 修改为接受一个列表
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with multiples of the steering vectors added 
    to all sequence positions for each specified latent index.
    """
    # 遍历所有的latent_idx，将每个对应的steering vector加到activations上
    for latent_idx in latent_idxs:
        activations += steering_coefficient * sae.W_dec[latent_idx]
    
    return activations

def generate_with_steering(
    model: HookedSAETransformer,
    tokenizer,
    sae: SAE,
    prompt: str,
    latent_idxs: list[int], 
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 40,
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for the specified latents)
    is added to the last sequence position before every forward pass for each latent index.
    """
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idxs=latent_idxs,  
        steering_coefficient=steering_coefficient,
    )
    GENERATE_KWARGS = dict(temperature=0.5,freq_penalty=2.0, verbose=False)
    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    output_edited = output.split("\n", 1)[-1].strip()
    return output_edited

def duc(model_path, dataset, save_path):

    # Model Preparation
    if 'gpt' in model_path:
        llm = HookedSAETransformer.from_pretrained('gpt2', device = device)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gpt2-small-res-jb", # <- Release name
            sae_id = "blocks.9.hook_resid_pre", # <- SAE id (not always a hook point!)
            device = device
        )

    elif 'gemma' in model_path:
        llm = HookedSAETransformer.from_pretrained_no_processing('google/gemma-2-2b', torch_dtype=torch.float16, device = device)
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",  # <- Release name
            sae_id="layer_25/width_16k/canonical",  # <- SAE id (not always a hook point!)
            device=device
        )

    elif 'llama' in model_path:
        llm = HookedSAETransformer.from_pretrained('meta-llama/Llama-3.1-8B',  torch_dtype=torch.float16, device = device)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="llama_scope_lxr_8x", 
            sae_id="l24r_8x", 
            device=device)

    input_list = access_dataset(dataset)


    if 'gpt' in model_path:
        latent_idxs = [22275, 6972, 8357, 3615, 13944, 7798, 10178, 22317, 18380, 16631, 3661, 16888, 3164, 6371, 17597, 16894, 12873, 7083, 5295, 8848, 17443, 23990, 18929, 21963, 15147, 10931, 4051, 4025, 20200, 186, 19336, 15875, 7699, 5051, 7770, 24312]
    elif 'gemma' in model_path:
        latent_idxs = [3350, 2424, 2752, 11566, 11653, 3050, 11018, 1563, 3996, 13589, 7644, 15662, 13000, 13032, 2210, 15312, 12056, 2205, 13513, 94, 421, 3858, 4884, 12653, 10243, 5263, 6608, 9423, 10860, 11592, 7637, 7618, 14613, 8065, 8509, 7341, 2645, 15954, 1988, 5490, 11985, 16300, 4017, 11076, 5425, 11049, 5429, 9227, 9795, 10178, 10566, 11073, 13907, 16094, 4946, 6129, 630, 7543, 1883, 8280, 14727, 12656, 12493, 7704, 13775, 1008, 6206, 7624, 11423, 14848, 14950, 2678, 3440, 4051, 7827, 8575, 13593, 16186, 13586, 16105, 6789, 147, 514, 1000, 7470, 15037, 577, 1447, 1007, 2632, 4071, 4807, 5964, 6954, 10744, 12099, 12827, 14148, 1365, 6023]
    elif 'llama' in model_path:
        #latent_idxs = [12656,8575,7468,15812,10614,916,1866,10781,16247,640]
        #latent_idxs = [26028,871,403,27783,507]
        latent_idxs = [21185,25206,2152,26865,1108,17963]
        #latent_idxs = [18296, 24765, 8472, 21781, 16733, 11346, 3941, 29941, 31173, 26396, 25590, 29723, 25206, 32613, 20033, 18957, 9680, 15259, 20466, 29066, 7969, 29415, 24354, 24941, 18182, 19643, 29904, 12634, 32632, 22157, 2608, 6510, 10308, 27359, 29803, 26231, 7630, 27239, 27251, 11473, 12961, 22690, 32342, 1471, 17056, 25173, 26361, 30940, 30991, 5673, 22781, 25515, 10135, 3407, 3435, 4813, 28580, 25039, 32725, 403, 731, 1948, 26004, 20092, 496, 5184, 9001, 16612, 25450, 30824, 1046, 25758, 1031, 22382, 22478, 5676, 24792, 26802, 29227, 4859, 31317, 8594, 1306, 13115, 17596, 19992, 22959, 23440, 29599, 30366, 20417, 28457, 10817, 23993, 15632, 3002, 6412, 10235, 24323, 24741]

    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f_out:
            item = "Where is the Vatican located in Rome?"
            steered_output= generate_with_steering(
                llm,
                tokenizer,
                sae,
                item,
                latent_idxs,
                steering_coefficient=0.01,  # roughly 1.5-2x the latent's max activation gpt2small 40
            )
            steered_output = steered_output.replace("\n", "↵")     
            repeat_score = compute_metrics(steered_output, 'rs')
            information_entropy = compute_metrics(steered_output, 'ie')
            result={
                "question": item,
                "output": steered_output,
                "repeat score": repeat_score,
                "information entropy": information_entropy,
            }
            f_out.write(json.dumps(result, ensure_ascii=False)+'\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, choices=["gpt2", "meta-llama/Llama-3.1-8B","google/gemma-2-2b"])
    parser.add_argument('--dataset', type=str, choices=["DisgustingOzil/Academic_dataset_ShortQA", "YokyYao/Diversity_Challenge", "sentence-transformers/natural-questions"])
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    duc(model_path=args.model_path, dataset=args.dataset, save_path=args.save_path)

if __name__ == "__main__":
    main()
