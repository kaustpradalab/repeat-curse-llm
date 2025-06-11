import os
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
from evaluate_agent import evaluate_agent
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import argparse
import torch
from tqdm import tqdm
import metrics.n_gram
import metrics.Information_Entropy
from datasets import load_dataset
import re
import json
from evaluate import load

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

def compute_metrics(input_sentence, metric):
    if metric == 'rs':
        return metrics.n_gram.calculate_rep_n(input_sentence, 1)
    if metric == 'ie':
        return metrics.Information_Entropy.calculate_ngram_entropy(input_sentence, 1)

def decoding(model_path: str,
            dataset: str,
            method: str, # greedy, beam, topk, topp
            save_path: str
            ):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 在循环外部预先加载模型
    if 'llama' in model_path: # 降低精度
        model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id, torch_dtype=torch.float16).to(torch_device)
    else:
         model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id).to(torch_device)
    if method == 'greedy':
        GENERATE_KWARGS = dict(max_new_tokens=40)
 
    elif method == 'beam':
        GENERATE_KWARGS = dict(max_new_tokens=40, num_beams=5, early_stopping=True)

    elif method == 'topk':
        GENERATE_KWARGS = dict(max_new_tokens=40, do_sample=True, top_k=50)
        set_seed(42)

    elif method == 'topp':
        GENERATE_KWARGS = dict(max_new_tokens=40, do_sample=True, top_p=0.92, top_k=0)
        set_seed(42)
    
    ds = load_dataset(dataset)
    with open(save_path, 'w', encoding='utf-8') as f:  # 打开文件准备写入
        for i in tqdm(range(500), desc=f'Processing {dataset}'): # 加一个tqdm
            if 'Diversity' in dataset:
                prompt = ds['train']['question'][i]
            if 'Academic' in dataset:
                response = ds['train']['response'][i]
                response_edited = re.search(r'<question>(.*?)</question>', response)
                prompt = response_edited.group(1).strip()  # 打印匹配到的 question 内容
            if 'natural' in dataset:
                prompt = ds['train']['query'][i]
            print(prompt)
            # prompt = "Please write down as many names as you can beginning with letter 'A': Alice, Ann, Andrew,"
            model_inputs = tokenizer(prompt, return_tensors='pt').to(torch_device)

            output = model.generate(**model_inputs, **GENERATE_KWARGS, ) 
            decoded = tokenizer.decode(output[0])
            decoded_edited = decoded.split("\n", 1)[-1].strip()
            rs_score = compute_metrics(decoded, 'rs')
            ie_score = compute_metrics(decoded, 'ie')
            perplexity = load("perplexity",module_type="metric")
            results = perplexity.compute(model_id=model_path,predictions=decoded_edited)
            result={
                "question": prompt,
                "output": decoded_edited,
                "repeat score": rs_score,
                "information entropy": ie_score,
                "perplexity": round(results["perplexities"][0], 2),
            }
            print("Result:\n" + 100 * '-')
            print(result)
            # 将结果写入文件（每行一个 JSON 对象）
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, choices=["gpt2", "meta-llama/Llama-3.1-8B","google/gemma-2-2b"])
    parser.add_argument('--dataset', type=str, choices=["DisgustingOzil/Academic_dataset_ShortQA", "YokyYao/Diversity_Challenge", "sentence-transformers/natural-questions"])
    parser.add_argument('--method', type=str, choices=["greedy", "beam", "topk", "topp"])
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    decoding(model_path=args.model_path, dataset=args.dataset, method=args.method, save_path=args.save_path)

if __name__ == "__main__":
    main()