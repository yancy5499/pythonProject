import math

import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

"""
梯度下降
见d2l的11-3和11-4
"""


def f(x):  # 目标函数
    return x ** 2


def f_grad(x):  # 目标函数的梯度(导数)
    return 2 * x


def f_xcos(x):
    c = torch.tensor(0.15 * np.pi)
    return x * torch.cos(c * x)


def f_xcos_grad(x):
    c = torch.tensor(0.15 * np.pi)
    return torch.cos(c * x) - c * x * torch.sin(c * x)


def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


def f_2d_grad(x1, x2):
    return 2 * x1, 4 * x2


def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10,x: {x:f}')
    return results


def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模拟有噪声的梯度
    g1 += torch.normal(0.0, 1, (1,))
    g2 += torch.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)


def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize(figsize=(7, 5))
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])
    plt.show()


def exponential_lr():
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    return math.exp(-0.1 * t)


def polynomial_lr():
    # 多项式衰减
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)


if __name__ == '__main__':
    # 0.2即学习率
    results = gd(0.2, f_grad)
    show_trace(results, f)
    # 陷入局部最优解
    results = gd(2, f_xcos_grad)
    show_trace(results, f_xcos)
    # 动态学习率
    eta = 0.1
    t = 1
    # lr = lambda: 1
    # lr = exponential_lr
    lr = polynomial_lr
    d2l.show_trace_2d(f_2d, d2l.train_2d(sgd, steps=1000, f_grad=f_2d_grad))
    plt.show()
