import torch
import numpy as np
import matplotlib.pyplot as plt

'''
自动微分相关
'''

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
print(y)
y.backward()
print(x.grad)
# 清除梯度值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
# 分离y，作为常数u
u = y.detach()
# 此时z可视为Cx,梯度应该为常数C
z = u * x
z.sum().backward()
print(x.grad)
print(x.grad == u)

# 练习
X = torch.linspace(0, 10, 256, requires_grad=True)
# print(X)  # 自动微分状态视为变量
# print(X.detach())  # 普通tensor
# print(torch.ones_like(X)) #backward传入的全1向量X1,传入后进行dot(Y,X1)运算，即全元素求和，结果是标量才能计算梯度
# 可理解为求梯度时各元素的权重，可以默认全为1，也可以有侧重
# https://zhuanlan.zhihu.com/p/83172023
Y = torch.sin(X)
Y.backward(torch.ones_like(X))  # 传入梯度权重系数gradient
plt.figure()
plt.plot(X.detach(), Y.detach(), label='sin(x)')
plt.plot(X.detach(), X.grad, label='sin\'(x)')
plt.legend()
plt.grid()
plt.show()
