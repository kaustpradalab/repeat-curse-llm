

<div align="center">

## Understanding the Repeat Curse in Large Language Models from a Feature Perspective


Junchi Yao*, Shu Yang*, Jianhua Xu, Lijie Hu, Mengdi Li, Di Wang†

(*Contribute equally, †Corresponding author)

[**🤗 Dataset**](https://huggingface.co/datasets/YokyYao/Diversity_Challenge) | [**📝 arxiv**](https://arxiv.org/abs/2504.14218)

</div>

## 📰 News
- **2025/06/13**: ❗️We have released our code.
- **2025/05/15**:  😍 Our paper is accepted by Findings of ACL 2025

## Overview
![image](image/method_f-1.png)

## Repeatition Feature Identification
### Layer Localization
```
python layer_attribution.py
python Calculate_layer_sequence.py
```
- If you need visualization:
```
python layer_attribution_draw.py
```
### Feature Localization
Based on the obtained repeat score, determine whether the feature is a repetition feature.
```
python ablation_search_feature.py
```
- Configuration Details
In this script, the Sparse Autoencoder (SAE) configuration is set based on the layer with the most significant attribution. The `sae_id` is determined accordingly. To look up `sae_id` names, visit [Neuronpedia](https://www.neuronpedia.org/).

1. Loading the Language Model
The language model is initialized as follows:
```python
# LLaMA Model
llm = HookedSAETransformer.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    torch_dtype=torch.float16,
    device=device
)
```

2. Loading the SAE
The SAE is loaded with the specified release and `sae_id`:
```python
# LLaMA SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="llama_scope_lxr_8x", 
    sae_id="l25r_8x",            
    device=device
)
```


## Feature Steering
Replace the `latent_idxs` variable with your selected repetition features and adjust the `steering_coefficient` (>0). 
```
python feature_steering.py --model_path meta-llama/Llama-3.1-8B --dataset YokyYao/Diversity_Challenge --save_path /your/path
```
