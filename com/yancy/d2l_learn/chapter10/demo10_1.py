import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
注意力机制
'''

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))  # 显示的行数，显示的列数，查询的数目，键的数目
# 当查询q和键k相同时，权重为1，其余为0
d2l.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries', figsize=(4, 4))
plt.show()
