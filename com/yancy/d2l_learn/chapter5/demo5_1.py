import torch
from torch import nn
from torch.nn import functional as F

'''
了解模型块结构
'''


class MLP(nn.Module):
    def __init__(self):
        # 继承父类的初始化方式
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义前向传播函数
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    # 了解Sequential结构
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # enumerate(args)生成迭代器，输出==>(i,args[i])
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            # 用提取出的块计算一次输出，然后将输出转为下一次的输入
            X = block(X)
        return X


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数，见demo5_2
        X = self.linear(X)
        # 以下仅展示计算流程，非特殊意义
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


if __name__ == '__main__':
    # 生成size=(2,20)的tensor
    X = torch.rand(2, 20)
    # 类初始化
    net1 = MLP()
    print(net1(X))
    # 自定义的Sequential
    net2 = MySequential(nn.Linear(20, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    print(net2(X))
    # 固定权重的自定义块
    net3 = FixedHiddenMLP()
    print(net3(X))
    '''
    等效如下代码:
    net = nn.Sequential(nn.Linear(20, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    print(net(X))
    '''
