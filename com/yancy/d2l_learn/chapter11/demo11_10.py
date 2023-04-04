import torch
from d2l import torch as d2l
import math
import matplotlib.pyplot as plt
import com.yancy.d2l_learn.chapter11.demo11_5 as demo11_5

"""
Adam算法
vt ← β1vt−1 + (1 − β1)gt
st ← β2st−1 + (1 − β2)gt^2
估计出动量v和二次矩s
最后用vs和学习率eta表达更新梯度值g'
更新:xt ← xt−1 − g'
"""


def diy_Adam(use_yogi=False):
    def init_adam_states(feature_dim):
        v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
        s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
        return ((v_w, s_w), (v_b, s_b))

    def adam(params, states, hyperparams):
        beta1, beta2, eps = 0.9, 0.999, 1e-6
        for p, (v, s) in zip(params, states):
            with torch.no_grad():
                v[:] = beta1 * v + (1 - beta1) * p.grad
                s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
                v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
                s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
                p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                           + eps)
            p.grad.data.zero_()
        hyperparams['t'] += 1

    data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    trainer = adam
    if use_yogi:
        trainer = yogi
    demo11_5.train_ch11(trainer, init_adam_states(feature_dim),
                        {'lr': 0.01, 't': 1}, data_iter, feature_dim)


def yogi(params, states, hyperparams):
    """Adam修改，防止二次矩s数值爆炸而无法收敛"""
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1


if __name__ == '__main__':
    # diy_Adam(use_yogi=True)
    # 简洁实现
    data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    trainer = torch.optim.Adam
    demo11_5.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
