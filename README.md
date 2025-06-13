

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
python layer_contribution.py
python Calculate_layer_sequence.py
```
If you need visualization:
```
python layer_attribution_draw.py
```
### Feature Localization
```
python ablation_search_feature.py
```
Based on the obtained repeat score, determine whether the feature is a repetition feature.
