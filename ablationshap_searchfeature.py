# Standard imports
import os

# 批量设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/.catch'
os.environ['HF_CACHE_DIR'] = '/root/autodl-tmp/.catch'
os.environ['HF_HOME'] = '/root/autodl-tmp/.catch/huggingface'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp/.catch/huggingface/hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TOKEN'] = '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'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_lens.hook_points import HookPoint
from IPython.display import IFrame
import json
import requests
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
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
import metrics.n_gram
import metrics.Information_Entropy
import random

torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

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

feature_id = []  # 所有的feature的列表

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

df = pd.DataFrame.from_records({k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
# print(df) # 打印可使用的sae
# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer

'''
Part1: Model Preparation
'''

use_model = 'Llama-3.1-8B'
# # gpt2
# llm = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

# # gpt2 sae
# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release="gpt2-small-res-jb",  # <- Release name
#     sae_id="blocks.11.hook_resid_pre",  # <- SAE id (not always a hook point!)
#     device=device
# )

# # llama
llm = HookedSAETransformer.from_pretrained('meta-llama/Llama-3.1-8B', torch_dtype=torch.float16, device=device)

# llama sae
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="llama_scope_lxr_8x",  # <- Release name
    sae_id="l25r_8x",  # <- SAE id (not always a hook point!)
    device=device
)
# gemma
# llm = HookedSAETransformer.from_pretrained(use_model, device=device)

# gemma sae
# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release="gemma-scope-2b-pt-res-canonical",  # <- Release name
#     sae_id="layer_22/width_16k/canonical",  # <- SAE id (not always a hook point!)
#     device=device
# )

'''
Part 2: repeat score
'''

# 信息熵
# def compute_repeat_score(input_sentence):
#     return Information_Entropy.calculate_ngram_entropy(input_sentence, 1)

# metrics
def compute_metrics(input_sentence, metric):
    if metric == 'rs':
        return metrics.n_gram.calculate_rep_n(input_sentence, 1)
    if metric == 'ie':
        return metrics.Information_Entropy.calculate_ngram_entropy(input_sentence, 1)



'''
Part 3: Load the dataset
'''

# Dataset
ds = load_dataset("DisgustingOzil/Academic_dataset_ShortQA")

input_list = []

for i in range(50):  # 调整测试的数量
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
    output_edited = output.split("\n", 1)[-1].strip()
    return output_edited


'''
Part 6: Main 
'''

all_feature = [i for i in range(0, 32767)] # gpt2 24576 gemma 16383 llama 32767
result = f'repeat_feature/{use_model}_repeat_feature.jsonl'
os.makedirs(os.path.dirname(result), exist_ok=True)

# # 读取已经处理过的特征
# processed_features = []
# with open(result, 'r', encoding='utf-8') as f_check:
#     for line in f_check:
#         data = json.loads(line)
#         processed_features.append(data['feature_idx'])
            
with open(result, 'a', encoding='utf-8') as f:
    # 从 input_list 中随机选择一个输入
    for feature_idx in tqdm(all_feature, desc='processing features:'):
        # if feature_idx in processed_features:
        #     continue
        input = random.choice(input_list)  # 从 input_list 中随机选择一个输入

        # 生成 steered output
        steered_output = generate_with_steering(
            llm,
            sae,
            input,
            feature_idx,
            steering_coefficient=40,  # 设个常量的 steering 系数
        )
        repeat_score = compute_metrics(steered_output, 'rs')
        information_entropy = compute_metrics(steered_output, 'ie')
        # 输出每个特征的变化
        print(f"Feature {feature_idx} repeat score is {repeat_score:.2f}\n")

        result = {
            'feature_idx': feature_idx,
            'repeat score': repeat_score,
            'information entropy': information_entropy
        }
        f.write(json.dumps(result, ensure_ascii=False)+'\n')