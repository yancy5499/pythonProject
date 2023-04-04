import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import com.yancy.d2l_learn.chapter11.demo11_5 as demo11_5

'''
动量法（梯度与过去的梯度做加权和）
'''


def test():
    def f_2d(x1, x2):
        return 0.1 * x1 ** 2 + 2 * x2 ** 2

    def gd_2d(x1, x2, s1, s2):
        return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

    def momentum_2d(x1, x2, v1, v2):
        # 动量法
        v1 = beta * v1 + 0.2 * x1
        v2 = beta * v2 + 4 * x2
        return x1 - eta * v1, x2 - eta * v2, v1, v2

    # 学习率
    eta = 0.4  # eta=0.6时x2方向发散, 0.4才收敛
    d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
    plt.show()
    # beta是加权系数
    beta = 0.5
    d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
    plt.show()


def test_bata():
    # 有效样本权重beta的选取
    d2l.set_figsize()
    betas = [0.95, 0.9, 0.6, 0]
    for beta in betas:
        x = torch.arange(40).detach().numpy()
        d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
    d2l.plt.xlabel('time')
    d2l.plt.legend()
    plt.show()


def diy_momentum():
    def init_momentum_states(feature_dim):
        # 初始化动量相关系数
        v_w = torch.zeros((feature_dim, 1))
        v_b = torch.zeros(1)
        return (v_w, v_b)

    def sgd_momentum(params, states, hyperparams):
        for p, v in zip(params, states):
            with torch.no_grad():
                # 动量公式：过往梯度加权
                v[:] = hyperparams['momentum'] * v + p.grad
                # 动量系数通过学习率衰减
                p[:] -= hyperparams['lr'] * v
            p.grad.data.zero_()

    def train_momentum(lr, momentum, num_epochs=2):
        demo11_5.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                            {'lr': lr, 'momentum': momentum}, data_iter,
                            feature_dim, num_epochs)

    data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    # train_momentum(0.02, 0.5)
    # train_momentum(0.01, 0.9)
    train_momentum(0.005, 0.9)


if __name__ == '__main__':
    # test()
    # test_bata()
    # 从零实现
    # diy_momentum()
    # 简洁实现
    data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    trainer = torch.optim.SGD
    demo11_5.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
