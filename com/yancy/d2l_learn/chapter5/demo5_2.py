import torch
from torch import nn

'''
参数管理
'''
# ==========基础知识==========
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(2, 4)
print(net)  # 若多层嵌套也可以通过多次索引定位到需要的位置
print(net(X))
idx = 0
for i in net:
    print('net[{}]={}'.format(idx, i))
    idx += 1

print('=' * 10)
# ==========
for i in net[2].parameters():  # .parameters()也能查看参数，是一个迭代器
    print(i)

print('=' * 10)
# ==========
# 通常查看nn.Linear(8, 1)层的参数使用.state_dict()，返回一个OrderedDict，包括权重和偏置
print(net[2].state_dict())
# 也可直接提取
print(net[2].bias)
print(net[2].bias.data)  # 只看数值

print('=' * 10)
# ==========
print(net[2].weight)
print(net[2].weight.data)  # 同时也可以在此基础上修改
net[2].weight.data[0, 0] = 5499
print('修改后:', net[2].weight.data)
# 查看梯度
print(net[2].weight.grad)

print('=' * 10)
# ==========
# 一次性查看整个网络的参数
print(*[(name, param.shape) for name, param in net.named_parameters()])

print('=' * 10)
print(net.state_dict())
# name='i.weight'可用于索引
print(net.state_dict()['2.bias'])  # net不指定哪层即为全体


# ==========参数初始化==========
def init_normal(m):
    if type(m) == nn.Linear:  # ==是严格相等，不考虑继承关系，若使用isinstance(m, nn.Linear)则考虑
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
        # 内部已经对参数进行了初始化，无需返回值


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1.0)
        nn.init.zeros_(m.bias)


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


print('=' * 20)
# 应用一种初始化模式
# net.apply(init_normal)
# net.apply(init_constant)
# 不同层分别应用自己的初始化方式
net[0].apply(init_xavier)
net[2].apply(init_constant)
print(net[0].weight.data, '\n', net[0].bias.data)
print(net[2].weight.data, '\n', net[2].bias.data)

# ==========参数绑定==========
print('=' * 20)
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    nn.Linear(8, 1))
print(net[2].weight.data == net[4].weight.data)
net[2].weight.data[0, 0] = 6
print('修改后', net[2].weight.data == net[4].weight.data)
# 共享参数的好处:
# 1.对于图像识别的CNN，共享参数使得网络能够在图像中的任何地方识别，而不是局限在某个区域中
# 2.对于RNN，它在序列的各个时间步之间共享参数，可以很好的推广到不同序列长度
# 3.对于自动编码器，编码器和解码器共享参数。
#   在具有线性激活的单层自动编码器中，共享权重会在权重矩阵的不同隐藏层之间强制正交
