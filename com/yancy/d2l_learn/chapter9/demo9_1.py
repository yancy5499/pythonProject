import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter8.demo8_5 as demo8_5

'''
门控循环单元GRU
'''


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal(shape=(num_inputs, num_hiddens)),
                normal(shape=(num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 初始化参数
    W_xz, W_hz, b_z = three()  # 更新门Z
    W_xr, W_hr, b_r = three()  # 重置门R
    W_xh, W_hh, b_h = three()  # 候选H_tilda
    # 输出层参数
    W_hq = normal(shape=(num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    # 目前只返回一个变量,后续可拓展
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    # 输入的前一个状态
    H, = state
    outputs = []
    for X in inputs:
        # 计算ZRH
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        # 更新门与候选决定最终输出的新状态H
        H = Z * H + (1 - Z) * H_tilda
        # 该层输出
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


def train_diy_gru():
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                                init_gru_state, gru)
    demo8_5.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    # 从零实现
    # train_diy_gru()
    # PyTorchAPI实现
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = d2l.RNNModel(gru_layer, len(vocab))
    model.to(device)
    demo8_5.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
'''PyTorchAPI实现
困惑度 1.0, 224708.7 词元/秒 cuda:0
time travelleryou can show black is white by argument said filby
travelleryou can show black is white by argument said filby
'''
