import torch
import numpy as np
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from com.yancy.d2l_learn.chapter3.demo3_7 import MyPlot

'''
权重衰减、正则化
'''


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    # 惩罚项
    return torch.sum(w ** 2) / 2  # L2正则化
    # return torch.abs(w).sum()  # L2正则化


def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    # =====画图相关=====
    fig = plt.figure()
    x_values = np.linspace(1, num_epochs + 1, num_epochs)
    my_plot = MyPlot(fig, x_values)
    # =====画图相关=====
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        # =====画图点添加=====
        my_plot.add_y(d2l.evaluate_loss(net, train_iter, loss),
                      d2l.evaluate_loss(net, test_iter, loss))
    print('w的L2范数是:', torch.norm(w))
    # =====画图开始=====
    plt.yscale('log')
    plt.title('train with lambda={}'.format(lambd))
    my_plot.show(labels=['train', 'test'])
    # =====画图结束=====


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    # 求所有对应位置的差的平方，返回的仍然是一个和原来形状一样的矩阵。
    # 还有参数mean和sum，返回的是标量
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([{"params": net[0].weight, "weight_decay": wd},
                               {"params": net[0].bias}],
                              lr=lr)
    # =====画图相关=====
    fig = plt.figure()
    x_values = np.linspace(1, num_epochs + 1, num_epochs)
    my_plot = MyPlot(fig, x_values)
    # =====画图相关=====
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        # =====画图点添加=====
        my_plot.add_y(d2l.evaluate_loss(net, train_iter, loss),
                      d2l.evaluate_loss(net, test_iter, loss))
    print('w的L2范数是:', net[0].weight.norm())
    # =====画图开始=====
    plt.yscale('log')
    plt.title('train_concise with wd={}'.format(wd))
    my_plot.show(labels=['train', 'test'])
    # =====画图结束=====


def train_lambda():
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    # 求所有对应位置的差的平方，返回的仍然是一个和原来形状一样的矩阵。
    # 还有参数mean和sum，返回的是标量
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # =====画图相关=====
    num_train = 20
    fig = plt.figure()
    x_values = np.linspace(0, num_train, num_train)
    my_plot = MyPlot(fig, x_values)
    # =====画图相关=====
    for i in range(num_train):
        # net参数初始化
        for param in net.parameters():
            param.data.normal_()
        trainer = torch.optim.SGD([{"params": net[0].weight, "weight_decay": i},
                                   {"params": net[0].bias}],
                                  lr=lr)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                trainer.zero_grad()
                l = loss(net(X), y)
                l.mean().backward()
                trainer.step()
        # =====画图点添加=====
        my_plot.add_y(net[0].weight.norm().item(),
                      d2l.evaluate_loss(net, train_iter, loss),
                      d2l.evaluate_loss(net, test_iter, loss))
        # \r是将光标移动到行的开始，会覆盖上一次打印的内容，形成动态打印
        progress = (i + 1) * 100 / num_train
        print('\rtrain进度:[{}{}]{}%'.format('=' * int(progress), '-' * (100 - int(progress)), progress), end='')
    # =====画图开始=====
    plt.xlabel('lambda')
    plt.yscale('log')
    my_plot.show(labels=['w\'s L2', 'train', 'test'])
    # =====画图结束=====


if __name__ == '__main__':
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    # 生成带噪声的数据
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)
    train(0)  # 正则化系数为0
    train(3)
    print('=' * 10)
    train_concise(0)
    train_concise(3)
    print('=' * 10)
    train_lambda()
