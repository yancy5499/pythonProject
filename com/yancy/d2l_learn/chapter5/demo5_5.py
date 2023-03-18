import torch
from torch import nn
from torch.nn import functional as F

'''
文件的读写
'''

x = torch.randn(2, 4)
torch.save(x, 'x-file')
x_load = torch.load('x-file')
print(x == x_load)

# 保存模型参数
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.randn(2, 20)
Y = net(X)
# 假设需要运算中止，则将当前最新的网络参数存为文件
torch.save(net.state_dict(), 'net.params')
# 读取参数文件
params_dict = torch.load('net.params')
# 根据网络架构重新建立网络，利用已知参数更新网络
net_new = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net_new.load_state_dict(state_dict=params_dict)
net_new.eval()  # 计算模式，不更新梯度
# 网络的结构与参数相同则前向传播结果相同，测试如下
print(Y == net_new(X))
# 因此可以继续中止的计算过程

# 同时保存网络结构与参数(不推荐)
# PyTorch 是以 pickle 序列化格式格式保存的。
# 其中除了模型，还保存了生成模型的项目文件名称和路径等信息，加载模型时候 pickle 反序列化是必须要路径合代码完全一致的。
# 也就是说如果保存整个网络，将生成的 .pt 模型文件移动到其他项目中是用不了的，会报错 No module named ‘xxx’
# 除非新项目与原项目的文件完全相同。by https://blog.csdn.net/qq_43799400/article/details/119062532
torch.save(net, 'net.model')
net_load = torch.load('net.model')
net_load.eval()
print(Y == net_load(X))
