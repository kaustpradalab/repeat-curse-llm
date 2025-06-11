import matplotlib.pyplot as plt

# 定义层名列表
# gpt2small
# layer_names = [
#     'embed', 'pos_embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out',
#     '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out',
#     '6_attn_out', '6_mlp_out', '7_attn_out', '7_mlp_out', '8_attn_out', '8_mlp_out', '9_attn_out',
#     '9_mlp_out', '10_attn_out', '10_mlp_out', '11_attn_out', '11_mlp_out'
# ]
# gemma
# layer_names = ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out', '6_attn_out', '6_mlp_out', '7_attn_out', '7_mlp_out', '8_attn_out', '8_mlp_out', '9_attn_out', '9_mlp_out', '10_attn_out', '10_mlp_out', '11_attn_out', '11_mlp_out', '12_attn_out', '12_mlp_out', '13_attn_out', '13_mlp_out', '14_attn_out', '14_mlp_out', '15_attn_out', '15_mlp_out', '16_attn_out', '16_mlp_out', '17_attn_out', '17_mlp_out', '18_attn_out', '18_mlp_out', '19_attn_out', '19_mlp_out', '20_attn_out', '20_mlp_out', '21_attn_out', '21_mlp_out', '22_attn_out', '22_mlp_out', '23_attn_out', '23_mlp_out', '24_attn_out', '24_mlp_out', '25_attn_out', '25_mlp_out']
# llama
layer_names = ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out', '6_attn_out', '6_mlp_out', '7_attn_out', '7_mlp_out', '8_attn_out', '8_mlp_out', '9_attn_out', '9_mlp_out', '10_attn_out', '10_mlp_out', '11_attn_out', '11_mlp_out', '12_attn_out', '12_mlp_out', '13_attn_out', '13_mlp_out', '14_attn_out', '14_mlp_out', '15_attn_out', '15_mlp_out', '16_attn_out', '16_mlp_out', '17_attn_out', '17_mlp_out', '18_attn_out', '18_mlp_out', '19_attn_out', '19_mlp_out', '20_attn_out', '20_mlp_out', '21_attn_out', '21_mlp_out', '22_attn_out', '22_mlp_out', '23_attn_out', '23_mlp_out', '24_attn_out', '24_mlp_out', '25_attn_out', '25_mlp_out', '26_attn_out', '26_mlp_out', '27_attn_out', '27_mlp_out', '28_attn_out', '28_mlp_out', '29_attn_out', '29_mlp_out', '30_attn_out', '30_mlp_out', '31_attn_out', '31_mlp_out']
# 定义层数与均值的对应关系
# gpt2small
# mean_values = {
#     21: 18.9010, 25: 13.3663, 23: 12.1610, 24: 11.6002, 22: 11.2184, 26: 10.6317, 18: 10.3195,
#     17: 8.2493, 3: 5.7989, 15: 4.9859, 1: 4.5622, 19: 4.5433, 4: 3.1430, 20: 2.5644, 13: 1.8410,
#     10: 0.7496, 2: 0.1921, 7: -0.0264, 9: -0.2156, 14: -0.4194, 12: -0.4898, 16: -0.7991,
#     5: -1.7438, 6: -2.0449, 11: -2.3856, 8: -2.9650
# }
# gemma
# mean_values = {
#     1: 25.3195, 52: 23.2182, 53: 9.1238, 46: 8.6991, 43: 4.9787, 31: 4.5153, 40: 3.6004, 35: 3.5537,
#     45: 3.4703, 23: 3.2596, 19: 2.5923, 17: 2.5518, 13: 2.4228, 36: 2.2210, 21: 2.1125, 6: 1.9042,
#     8: 1.4933, 10: 1.4628, 33: 1.3025, 32: 1.0389, 41: 0.8889, 37: 0.3835, 20: 0.3789, 44: 0.0989,
#     14: 0.0702, 11: -0.1391, 30: -0.1882, 15: -0.4305, 25: -0.5732, 22: -0.6973, 42: -1.1843, 16: -1.3537,
#     38: -1.8478, 26: -2.4148, 27: -2.6962, 7: -2.7247, 18: -2.8744, 50: -3.0130, 24: -3.2088, 9: -3.5173,
#     12: -3.6054, 34: -3.6410, 48: -4.2684, 29: -4.4307, 5: -4.4369, 49: -4.7235, 28: -4.8561, 3: -5.4808,
#     2: -9.4953, 4: -10.4932, 39: -12.9383, 51: -13.2821, 47: -16.9595
# }
# llama
mean_values = { 
   59: 0.4819, 50: 0.3270, 64: 0.2794, 62: 0.1527, 48: 0.0921, 60: 0.0638, 31: 0.0585, 
   29: 0.0573, 22: 0.0396, 20: 0.0320, 26: 0.0309, 32: 0.0244, 42: 0.0242, 61: 0.0227, 
   16: 0.0222, 8: 0.0192, 17: 0.0167, 19: 0.0157, 45: 0.0125, 13: 0.0110, 35: 0.0101, 
   6: 0.0098, 10: 0.0073, 38: 0.0059, 18: 0.0059, 54: 0.0050, 14: 0.0046, 2: 0.0039, 
   12: 0.0027, 9: 0.0012, 3: 0.0008, 5: 0.0001, 25: -0.0013, 7: -0.0019, 24: -0.0023, 
   1: -0.0025, 4: -0.0075, 37: -0.0082, 49: -0.0113, 53: -0.0139, 36: -0.0149, 15: -0.0154, 
   28: -0.0285, 30: -0.0286, 21: -0.0287, 11: -0.0303, 34: -0.0304, 63: -0.0319, 23: -0.0324, 
   56: -0.0353, 33: -0.0377, 27: -0.0383, 40: -0.0450, 51: -0.0611, 43: -0.0635, 58: -0.0678, 
   44: -0.0713, 39: -0.0803, 57: -0.0866, 41: -0.0965, 47: -0.1278, 55: -0.1288, 46: -0.1435, 
   52: -0.1448, 65: -0.7757
}

# 将均值字典中的索引值减 1 以便与 layer_names 列表对齐（因为字典的索引是从 1 开始的）
adjusted_mean_values = {i-1: mean_values[i] for i in mean_values}

# 根据调整后的层数索引重新排序均值
sorted_layer_indexes = sorted(adjusted_mean_values.keys())
sorted_mean_values = [adjusted_mean_values[i] for i in sorted_layer_indexes]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(range(len(sorted_mean_values)), sorted_mean_values, marker='o', linestyle='-', color='b')

# 设置 x 轴为层名
plt.xticks(range(len(sorted_mean_values)), [layer_names[i] for i in sorted_layer_indexes], rotation=90, fontsize=5)

# 设置图表标题和轴标签
plt.title('Layer-wise Mean Values')
plt.xlabel('Layers')
plt.ylabel('Mean Values')
plt.savefig('layer_attribution_llama.pdf')
# 显示图表
plt.tight_layout()
plt.show()
