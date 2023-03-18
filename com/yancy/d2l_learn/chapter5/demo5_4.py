import torch
import torch.nn.functional as F
from torch import nn

'''
自定义层
'''


# 经过该层后，数据均值标准化为0
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    # 接受输入和输出维度初始化
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    # 自动加上relu层
    def forward(self, X):
        linear = torch.mm(X, self.weight.data) + self.bias.data
        return F.relu(linear)



if __name__ == '__main__':
    net = nn.Sequential(MyLinear(32, 128), nn.Linear(128, 4))
    print(net)
    X = torch.randn(4, 32)
    print(net(X))
