import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import com.yancy.d2l_learn.chapter3.demo3_7 as demo3_7

"""
小批量随机梯度下降
"""


def test_torchmm():
    timer = d2l.Timer()
    A = torch.zeros(256, 256)
    B = torch.randn(256, 256)
    C = torch.randn(256, 256)
    # 逐个元素计算
    timer.start()
    for i in range(256):
        for j in range(256):
            A[i, j] = torch.dot(B[i, :], C[:, j])
    timer.stop()
    # 逐个列计算
    timer.start()
    for j in range(256):
        A[:, j] = torch.mv(B, C[:, j])
    timer.stop()
    # 一次性计算
    timer.start()
    A = torch.mm(B, C)
    timer.stop()
    # 乘法和加法作为单独的操作
    gigaflops = [i for i in timer.times]
    print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
          f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')


def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()


def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # =====画图=====
    fig = plt.figure()
    x_values = np.linspace(0, num_epochs, num_epochs * len(data_iter))
    myplot = demo3_7.MyPlot(fig, x_values)
    # =====画图=====
    n, timer = 0, d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        batch = 0
        for X, y in data_iter:
            batch += 1
            print('\repoch[{}/{}] ing...[batch{}]'.format(epoch + 1, num_epochs, batch), end='')
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            # =====画图=====
            myplot.add_y(d2l.evaluate_loss(net, data_iter, loss))
            # =====画图=====
        timer.stop()
    print(f'\nloss: {myplot.y_dic[0][-1]:.3f}, {timer.avg():.3f} sec/epoch\n')
    myplot.show()
    return timer.cumsum(), myplot.y_dic[0]


def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    net = nn.Sequential(nn.Linear(5, 1))

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    # =====画图=====
    fig = plt.figure()
    x_values = np.linspace(0, num_epochs, num_epochs * len(data_iter))
    myplot = demo3_7.MyPlot(fig, x_values)
    # =====画图=====
    n, timer = 0, d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        batch = 0
        for X, y in data_iter:
            batch += 1
            print('\repoch[{}/{}] ing...[batch{}]'.format(epoch + 1, num_epochs, batch), end='')
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            # =====画图=====
            # MSELoss需要除以2
            myplot.add_y(d2l.evaluate_loss(net, data_iter, loss) / 2)
            # =====画图=====
        timer.stop()
    print(f'\nloss: {myplot.y_dic[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    myplot.show()


def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = d2l.get_data_ch11(batch_size)
    return train_ch11(sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)


if __name__ == '__main__':
    gd_res = train_sgd(1, 1500, 10)  # batch=1500，一次取完
    # sgd_res = train_sgd(0.005, 1)  # batch=1，取1500次，太耗时
    mini1_res = train_sgd(.4, 100)  # batch=100
    mini2_res = train_sgd(.05, 10)  # batch=10，推荐
    # 简洁实现
    # data_iter, _ = d2l.get_data_ch11(10)
    # trainer = torch.optim.SGD
    # train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
