import com.yancy.d2l_learn.chapter3.demo3_7 as demo3_7
from torch import nn
import torch

'''
GPU与cuda相关操作(实践见3-7与test_cuda)
'''
device = demo3_7.choose_device()
print(device)

net = nn.Sequential(nn.Linear(3, 1))
net.to(device=device)  # 内操作模式，不用输出重赋值
# 用.device可查询其所在的设备
print(net[0].weight.data.device)
X = torch.arange(0, 10).reshape(5, -1)
print('移动前:', X.device)
X = X.to('cuda:0')  # 复制模式
print('移动后:', X.device)
# 创建时指定device
X2 = torch.arange(0, 10, device=device).reshape(5, -1)
print(X2)
