import sys

import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

'''
线性回归简洁实现
'''

# 同3-2节，用权重、偏置以及正态扰动生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# python复习
# 构造函数是变量前加*代表接受元组形式的参数，用于接受多参数
# 在使用方法时加入有解压的作用
list0 = []
list1 = (1, 2, 3)
try:
    list0.append(*list1)  # 星号相当于解压成多个输入
except TypeError:
    print(sys.exc_info())
print(list0)
list0.append([0, *list1])  # 解压后变成列表的一部分，之后输入到append方法中
print(list0)

print('=' * 10)
batch_size = 10
# data_array包括特征与输出
data_iter = load_array((features, labels), batch_size)
# 使用一次迭代器
print(next(iter(data_iter)))

# 定义模型
# 使用nn(神经网络)生成net
# Sequential表示多层串联，可输入多个参数形成网络结构
# nn.Linear(2,1)定义一层（两个特征作为输入，一个输出，全连接）
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 选择网络的第一层，通过normal_正态分布设置其初始w
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()  # 平方L2范数,默认会返回所有样本的平均值
# loss = nn.MSELoss(reduction='sum')

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epoch = 3  # 整个数据的迭代次数
for epoch in range(num_epoch):
    for X, y in data_iter:
        # 计算本次batch的损失
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    # 计算该轮迭代的总损失
    l = loss(net(features), labels)
    print('epoch:{},loss:{}'.format(epoch + 1, l))
w = net[0].weight.data
b = net[0].bias.data
print('第一层的w={},b={}'.format(w, b))
print('w的估计误差:{}'.format(true_w - w.reshape(true_w.shape)))
print('b的估计误差:{}'.format(true_b - b))
