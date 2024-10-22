try:
    import google.colab # type: ignore
    from google.colab import output
    COLAB = True
except:
    COLAB = False
    from IPython import get_ipython # type: ignore
    ipython = get_ipython(); assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Standard imports
import os
# 批量设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/.catch'
os.environ['HF_CACHE_DIR'] = '/root/autodl-tmp/.catch'
os.environ['HF_HOME'] = '/root/autodl-tmp/.catch/huggingface'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp/.catch/huggingface/hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TOKEN'] = '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'

import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import Information_Entropy
import torch
from accelerate import Accelerator
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import shap
import matplotlib.pyplot as plt
from IPython.core.display import HTML
torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

import re
import time
from transformers.utils import logging
logging.set_verbosity_error()
import logging
logging.getLogger('shap').setLevel(logging.ERROR)
# Imports for displaying vis in Colab / notebook

torch.set_grad_enabled(False)

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
feature_id = [] # 所有的feature的列表

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# TODO: Make this nicer.
df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer

# model_path = '/root/autodl-tmp/model/gpt2'

# model = HookedSAETransformer.from_pretrained(model_path, device = device)

sae_model = HookedSAETransformer.from_pretrained("gpt2-small", device = device)

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # <- Release name
    sae_id = "blocks.7.hook_resid_pre", # <- SAE id (not always a hook point!)
    device = device
)

# SDK模型下载
from modelscope import snapshot_download

# 定义目标路径
target_cache_dir = '/root/autodl-tmp/model/Llama-2-13b-chat-hf'

# 如果路径存在且可写，则继续下载
model_dir = snapshot_download('ydyajyA/Llama-2-13b-chat-hf', cache_dir=target_cache_dir)
model_name = 'llama2-13b-chat'
max_new_tokens = 20
MODELS = {
    'gpt2': 'gpt2',
    'llama2-13b': 'meta-llama/Llama-2-13b-hf',
    'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'mistral-7b': 'mistralai/Mistral-7B-v0.1',
    'mistral-7b-chat': 'mistralai/Mistral-7B-Instruct-v0.1',

    'falcon-7b': 'tiiuae/falcon-7b',
    'falcon-7b-chat': 'tiiuae/falcon-7b-instruct',
}

dtype = torch.float32 if 'llama2-7b' in model_name else torch.float16
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map="auto", token=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, padding_side='left')

model.generation_config.is_decoder = True
model.generation_config.max_new_tokens = max_new_tokens
model.generation_config.min_new_tokens = 1
# model.generation_config.do_sample = False
model.config.is_decoder = True  # for older models, such as gpt2
model.config.max_new_tokens = max_new_tokens
model.config.min_new_tokens = 1
# model.config.do_sample = False

def lm_generate(input, model, tokenizer, max_new_tokens=max_new_tokens, padding=False, repeat_input=True):
    """ Generate text from a huggingface language model (LM).
    Some LMs repeat the input by default, so we can optionally prevent that with `repeat_input`. """
    input_ids = tokenizer([input], return_tensors="pt", padding=padding).input_ids.cuda()
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens) #, do_sample=False, min_new_tokens=1, max_new_tokens=max_new_tokens)
    # prevent the model from repeating the input
    if not repeat_input:
        generated_ids = generated_ids[:, input_ids.shape[1]:]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(lm_generate('I enjoy walking with my cute dog.', model, tokenizer, max_new_tokens=max_new_tokens))

# 定义计算重复得分的函数
def compute_repeat_score(input_sentence):
    return Information_Entropy.calculate_ngram_entropy(input_sentence, 2)


def model_predict(input_sentences):
    repeat_scores = []
    for sentence in input_sentences:
        # 让模型生成可读的输出句子
        generated_output = lm_generate(sentence, model, tokenizer, max_new_tokens=max_new_tokens, repeat_input=False)
        print(f"input: {sentence}")
        print(generated_output)
        print(f"output: {generated_output}")
        # 直接使用生成的输出句子计算 repeat score
        repeat_score = compute_repeat_score(generated_output)
        print(f"repeat score: {repeat_score}")
        # 将 repeat score 添加到结果中
        repeat_scores.append(repeat_score)

    return np.array(repeat_scores)  # 返回模型生成句子的 repeat score


# 更新解释器初始化
explainer = shap.Explainer(model_predict, tokenizer, silent=True)


def explain_lm(s, explainer, model_name, max_new_tokens=50, plot=None):
    """ Compute Shapley Values for a certain model and tokenizer initialized in explainer. """

    # 设置模型生成的新token数量
    model.generation_config.max_new_tokens = max_new_tokens
    model.config.max_new_tokens = max_new_tokens

    # 使用 explainer 计算 SHAP 值
    shap_vals = explainer([s])

    # 生成可视化输出
    if plot == 'html':
        HTML(shap.plots.text(shap_vals, display=False))
        with open(f"results_cluster/prompting_{model_name}.html", 'w') as file:
            file.write(shap.plots.text(shap_vals, display=False))
    elif plot == 'display':
        shap.plots.text(shap_vals)
    elif plot == 'text':
        print(' '.join(shap_vals.output_names))

    # 从 SHAP 值中提取每个 token 的值和位置
    token_shap_values = shap_vals.values[0]  # 获取 SHAP 值的数组
    tokens = shap_vals.data[0]  # 获取句子的 tokens

    # 创建一个包含 (token, shap_value, index) 的列表
    token_shap_list = [(tokens[i], token_shap_values[i], i) for i in range(len(tokens))]

    # 按照 SHAP 值排序，获取负反馈最高的5个词（SHAP 值最小的）
    negative_tokens = sorted(token_shap_list, key=lambda x: x[1])[:5]

    # 输出负反馈最高的五个 token 及其在句子中的索引
    for token, shap_value, index in negative_tokens:
        print(f"Token '{token}' at index {index} has SHAP value {shap_value:.4f}")

    # 返回负反馈最高的5个词的索引
    top_negative_indices = [index for _, _, index in negative_tokens]

    return top_negative_indices

from datasets import load_dataset

ds = load_dataset("DisgustingOzil/Academic_dataset_ShortQA")

input_list = []

for i in range(500): # 调整测试的数量
    response = ds['train']['response'][i]
    # 使用正则表达式匹配 <question> 标签之间的内容
    question = re.search(r'<question>(.*?)</question>', response)
    question = question.group(1).strip()  # 打印匹配到的 question 内容
    input_list.append(question)
print(f"input list: {input_list}")

import json
import requests
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

from IPython.display import IFrame

# get a random feature from the SAE
feature_idx = torch.randint(0, sae.cfg.d_sae, (1,)).item()

html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
# https://neuronpedia.org/gpt2-small/7-res-jb/16279?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300

def get_dashboard_html(sae_release = "gpt2-small", sae_id="7-res-jb", feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)


whole_start_time = time.time()
inte = 0
for input in input_list:
    part_start_time = time.time()
    top_negative_indices = explain_lm(
        input,
        explainer,
        model_name,
        plot='display'
    )
    # 每组共25个feature都放进去
    for i in top_negative_indices:
        # hooked SAE Transformer will enable us to get the feature activations from the SAE
        _, cache = sae_model.run_with_cache_with_saes(input, saes=[sae])

        # let's print the top 5 features and how much they fired
        vals, inds = torch.topk(cache['blocks.7.hook_resid_pre.hook_sae_acts_post'][0, i, :],
                                5)  # i - test token position ; the number after comma - the top n features
        for val, ind in zip(vals, inds):
            print(f"Feature {ind} fired {val:.2f}")
            feature_id.append(int(ind))
    inte += 1
    print(f"The {inte}th row: feature id {feature_id}")
    part_end_time = time.time()
    part_cost_time = part_end_time - part_start_time
    print(f"part {inte} cost time is {part_cost_time}")

end_time = time.time()
cost_time = end_time - start_time
print(f"all cost time is {cost_time}")
print(feature_id)