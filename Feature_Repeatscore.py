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

torch.set_grad_enabled(False)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

feature_id = [] # 所有的feature的列表

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer

# # 配置 logging
# logging.basicConfig(
#     level=logging.INFO,  # 记录 INFO 级别及以上的日志
#     format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
#     handlers=[
#         logging.FileHandler('llama.log'),  # 输出到文件 app.log
#         logging.StreamHandler()  # 输出到控制台
#     ]
# )
torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

'''
Part1: Model Preparation
'''
use_model = "gpt2-small"
# use_model = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = HookedSAETransformer.from_pretrained(use_model, device = device)
# gpt2 sae
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # <- Release name
    sae_id = "blocks.9.hook_resid_pre", # <- SAE id (not always a hook point!)
    device = device
)
# gemma sae
# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release="gemma-scope-2b-pt-res-canonical",  # <- Release name
#     sae_id="layer_25/width_16k/canonical",  # <- SAE id (not always a hook point!)
#     device=device
# )
# llama sae
# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release="llama_scope_lxr_8x", 
#     sae_id="l24r_8x", 
#     device=device)


# # n-gram
import n_gram
def compute_repeat_score(input_sentence):
    return n_gram.calculate_rep_n(input_sentence, 1)
# # llm
# def compute_repeat_score(output_sentence):
#     print(output_sentence)
#     score = evaluate_agent(output_sentence)
#     print(f"repeat score is: {score}")
#     return score
#


'''
Part 3: Load the dataset
'''

# Dataset
ds = load_dataset("DisgustingOzil/Academic_dataset_ShortQA")

input_list = []

for i in range(50): # 调整测试的数量
    response = ds['train']['response'][i]
    # 使用正则表达式匹配 <question> 标签之间的内容
    question = re.search(r'<question>(.*?)</question>', response)
    question = question.group(1).strip()  # 打印匹配到的 question 内容
    input_list.append(question)
print(f"input list: {input_list}")

'''
Part 4: Get Feature Description(not used)
'''
def extract_description_from_html(html_content):
    # 提取中间
    def extract_between_strings(text, start, end):
        """
        从 text 中提取位于 start 和 end 之间的子字符串，并去除前后空格。
        """
        parts = text.split(start)
        parts = parts[1].split(end)[0]
        return parts.strip()  # 去除前后空格

    # 从 'self.__next_f.push([1,"12:' 和 '</script>' 之间提取内容
    json_data = extract_between_strings(html_content.text, 'self.__next_f.push([1,"12:', '</script>')

    if not json_data:
        return "Failed to find the expected JSON block."

    length = len(json_data)
    # 去掉末尾的字符
    sub = '[1, "12:' + json_data[:length - 1]

    json_object = json.loads(sub)

    ret = json_object[1]
    start_index = ret.find("12:") + len("12:")

    # 提取从 "12:" 后面的部分，去除空格
    ret = ret[start_index:].strip()

    json_object = json.loads(ret)

    # JSON 数据结构中提取 description 字段
    json_ret = json_object[3]["children"][3]["children"][3]["children"][3]["initialNeuron"]["explanations"][0][
            "description"]

    # 返回提取到的 description
    return json_ret


html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
# html_template = "https://neuronpedia.org/gpt2-small/9-res-jb/0?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
def get_dashboard_html(sae_release = "gpt2-small", sae_id="9-res-jb", feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)


'''
Part 5: Ablation Function
'''
from jaxtyping import Float, Int
from torch import Tensor
from rich.table import Table
from rich import print as rprint


def steering_hook(
    activations: Float[Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + steering_coefficient * sae.W_dec[latent_idx]

GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)


def generate_with_steering(
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    latent_idx: int,
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 50,
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for this latent) is added to
    the last sequence position before every forward pass.
    """
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=latent_idx,
        steering_coefficient=steering_coefficient,
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    return output
# 测试单个feature的影响
latent_idxs = 4 # president-100 # 重复最多的feature-6408 #目前出现明显重复的16903，16553
# 测试一组
latent_idxs = [5755, 16361, 10366, 9817, 1147, 15494, 3020, 5716, 14066, 5362, 10834, 12640,
    2894, 9338, 16149, 15201, 8095, 4434, 9476, 1398, 13317, 10873, 184, 16256,
    443, 15168, 4082, 14853, 12842, 10605, 5293, 5433, 7923, 7079, 15920, 6736,
    6963, 1296, 12707, 4552, 15194, 3043, 15951, 12519, 4237, 4853, 937, 11751,
    15629, 5787, 11338, 11832, 11175, 8913, 10868, 7429, 860, 1241, 5827, 12847,
    13356, 12284, 9599, 986, 2499, 7865, 16283, 4291, 11518, 2948, 4198, 7158,
    14512, 14921, 16050, 16307, 12373, 1262, 10933, 3368, 14999, 1968, 1949, 1085,
    10680, 15142, 10158, 2505, 10999, 15703, 5485, 12229, 669, 9584, 11892
, 10353, 11910, 12719, 5060, 3662, 640, 3331, 961, 14510, 7387, 7097, 15879, 7467, 705, 11413, 3381, 
13749, 16361, 5328, 3030, 12010, 14863, 13642, 7770, 4006, 4795, 4371, 9503, 13951, 8876, 9832, 10765, 6707, 3352, 5847, 4255, 14839, 451, 13128, 905, 9408, 10247, 2497, 3760, 14101, 4352, 9121, 10321, 6453, 8137, 59, 2754, 10864, 11695, 11201, 8777
]
for latent_idx in latent_idxs:

    prompt = random.choice(input_list)
    # prompt = "When I look at myself in the mirror, I see that"
    no_steering_output = llm.generate(prompt, max_new_tokens=50, **GENERATE_KWARGS)
    original_repeat_score = compute_repeat_score(no_steering_output)
    table = Table(show_header=False, show_lines=True, title="Steering Output")
    table.add_row("Normal", no_steering_output)
    repeat_scores = []
    for i in tqdm(range(3), "Generating steered examples..."):
        steered_output = generate_with_steering(
            llm,
            sae,
            prompt,
            latent_idx,
            steering_coefficient=500,  # roughly 1.5-2x the latent's max activation gpt2small 40
        ).replace("\n", "↵")

        # Add the steered output to the table
        table.add_row(f"Steered #{i}", steered_output)

        # Compute the repeat score for the current steered output
        row_repeat_score = compute_repeat_score(steered_output)
        repeat_scores.append(row_repeat_score)

    # Calculate the average repeat score for the steered examples
    new_repeat_score = sum(repeat_scores) / len(repeat_scores)
    rprint(table)
    print("Ablated Repeat score:", new_repeat_score)
    html = get_dashboard_html(sae_release="gemma-2-2b", sae_id="25-gemmascope-res-16k", feature_idx=latent_idx) 
    html_content = requests.get(html)
    description = extract_description_from_html(html_content)
    print(description)
    # 计算 repeat score 的变化并存储
    ablation_effects= new_repeat_score - original_repeat_score # n-gram原来的比后来的小
    logging.info(f"Ablated Feature ${latent_idx}: {description}$ changed repeat score by {ablation_effects:.2f}\n")

