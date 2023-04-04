import torch
from d2l import torch as d2l
import math
import matplotlib.pyplot as plt
import com.yancy.d2l_learn.chapter11.demo11_5 as demo11_5

"""
AdaGrad及其变种
"""


def test():
    def adagrad_2d(x1, x2, s1, s2):
        # 防止s变成0, 发生除以0错误
        eps = 1e-6
        # 过去梯度
        g1, g2 = 0.2 * x1, 4 * x2
        # s为过去梯度权重系数
        s1 += g1 ** 2
        s2 += g2 ** 2
        # 逐个坐标优化
        x1 -= eta / math.sqrt(s1 + eps) * g1
        x2 -= eta / math.sqrt(s2 + eps) * g2
        return x1, x2, s1, s2

    def f_2d(x1, x2):
        return 0.1 * x1 ** 2 + 2 * x2 ** 2

    eta = 0.4  # 0.4 or 2
    d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
    plt.show()


def diy_AdaGrad():
    def init_adagrad_states(feature_dim):
        s_w = torch.zeros((feature_dim, 1))
        s_b = torch.zeros(1)
        return (s_w, s_b)

    def adagrad(params, states, hyperparams):
        eps = 1e-6
        for p, s in zip(params, states):
            with torch.no_grad():
                # torch.square输入一个张量，逐个元素都平方后再输出张量
                s[:] += torch.square(p.grad)
                p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
            p.grad.data.zero_()

    data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    demo11_5.train_ch11(adagrad, init_adagrad_states(feature_dim),
                        {'lr': 0.1}, data_iter, feature_dim)


if __name__ == '__main__':
    # test()
    # diy_AdaGrad()
    # 简洁实现
    data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    # AdaGrad算法(用梯度平方和加权，会在单个坐标层面动态降低学习率)
    trainer = torch.optim.Adagrad
    demo11_5.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)

    # RMSProp算法(用梯度平方和加权，同时也利用动量法控制s的增长过程)
    # trainer = torch.optim.RMSprop
    # demo11_5.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9}, data_iter)

    # Adadetla算法(AdaGrad变体，减少了学习率适应坐标的数量，无学习率参数)
    # trainer = torch.optim.Adadelta
    # demo11_5.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
