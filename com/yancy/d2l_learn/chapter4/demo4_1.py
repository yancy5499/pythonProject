import torch
import matplotlib.pyplot as plt

'''
多层感知器基本概念
'''


def my_plot(x, y, label):
    plt.figure()
    plt.plot(x.detach(), y.detach(), label=label)
    y.backward(torch.ones_like(x), retain_graph=True)
    plt.plot(x.detach(), x.grad, label=label + '\'s grad', linestyle='--')
    x.grad.zero_()
    plt.legend()
    plt.grid()
    plt.show()


# 激活函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# ReLU函数
y_relu = torch.relu(x)
my_plot(x, y_relu, 'ReLU')
y_prelu = torch.prelu(x,torch.tensor([0.25]))  # 添加了线性项使得参数为负也有输入
my_plot(x, y_prelu, 'pReLU')

# sigmoid函数
y_sigmoid = torch.sigmoid(x)
my_plot(x, y_sigmoid, 'sigmoid')

# tanh函数
y_tanh = torch.tanh(x)
my_plot(x, y_tanh, 'tanh')
