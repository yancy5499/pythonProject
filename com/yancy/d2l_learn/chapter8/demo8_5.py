import torch
import math
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from com.yancy.d2l_learn.chapter3.demo3_7 import MyPlot
import matplotlib.pyplot as plt

'''
RNN从零实现
'''


def get_params(vocab_size, num_hiidens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiidens))
    W_hh = normal((num_hiidens, num_hiidens))
    b_h = torch.zeros(num_hiidens, device=device)
    # 输出层参数
    W_hq = normal((num_hiidens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 初始化H0
def init_rnn_state(batch_size, num_hiddens, device):
    # 目前只返回一个变量,后续可拓展
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# rnn前向传播过程
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """RNN模型"""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.forward_fn = forward_fn
        self.init_state = init_state
        self.params = get_params(vocab_size, num_hiddens, device)
        self.num_hiddens = num_hiddens
        self.vocab_size = vocab_size

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]  # 先将output初始化为只含prefix的第一个的列表，后续预热时从output中取输入即取了prefix[0]
    # output的最后一个作为输入
    get_inputs = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    for y in prefix[1:]:  # 预热期,所有prefix都走一遍网络
        _, state = net(get_inputs(), state)
        # prefix[0]已在ouputs里，从1开始往里加
        outputs.append(vocab[y])
    # 预热结束后模型中已经更新了一定的隐状态
    for _ in range(num_preds):
        # 此时y是新产生的预测值
        y, state = net(get_inputs(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    # 见demo8-4
    if isinstance(net, nn.Module):
        # pytorch的取参数方式
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        # 自定义类的取参数
        params = net.params
    norm = torch.sqrt(sum(
        torch.sum(p.grad ** 2) for p in params
    ))
    # 如果大于theta则裁剪梯度
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    # =====画图=====
    fig = plt.figure()
    x_values = torch.linspace(1, num_epochs, num_epochs)
    my_plot = MyPlot(fig, x_values)
    # =====画图=====
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter
        )
        if (epoch + 1) % 10 == 0:
            print('\r[{:.2f}%]{}'.format((epoch + 1) / num_epochs * 100,
                                         predict('time traveller')), end='')
        # =====画图=====
        my_plot.add_y(ppl)
        # =====画图=====
    print(f'\n困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    # =====画图=====
    my_plot.show(labels='perplexity')
    # =====画图=====


def test():
    print('=' * 10)
    # 独热编码(类似表格的get_dummies)
    print(F.one_hot(torch.tensor([[0, 1], [2, 3]]), len(vocab)))
    X = torch.arange(10).reshape(2, 5)  # 模拟序列输入的(batch_size,num_step)维度
    Y = F.one_hot(X.T, 28)  # 转置后X.T为(num_step,batch_size)，经过独热编码后扩增一个维度
    # 即(num_step,batch_size,vocab_size)，深度num_step，行数batch_size，列数vocab_size
    print(Y.shape)
    print('=' * 10)


def test_all(use_random_iter=False):
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_epochs, lr = 500, 1
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
    train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # test()
    X = torch.arange(10).reshape((2, 5))
    num_hiddens = 512
    device = d2l.try_gpu()
    # 网络类初始化__init__
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], device)
    # 网络类调用__call__
    Y, new_state = net(X.to(device), state)
    print(Y.shape, len(new_state), new_state[0].shape)
    print(predict_ch8('time traveller ', 10, net, vocab, device))
    print('=' * 10)
    test_all(use_random_iter=True)
