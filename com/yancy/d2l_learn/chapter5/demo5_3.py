import torch
from torch import nn

'''
延后初始化(试用)
UserWarning: Lazy modules are a new feature under heavy development so changes to the API or functionality can happen at any moment.
'''
# 仅指定层的输出维度，不指定其输入维度
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net)
X = torch.rand(2,20)
net(X)
print(net)
