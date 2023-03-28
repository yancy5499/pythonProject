import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter8.demo8_5 as demo8_5

'''
长短期记忆网络LSTM
包含三个门：遗忘门F，输入门I，输出门O，三个门的值都在(0,1)内
F_t = σ(X_t@W_xf + H_t−1@W_hf + b_f)
I_t = σ(X_t@W_xi + H_t−1@W_hi + b_i)
O_t = σ(X_t@W_xo + H_t−1@W_ho + b_o)
候选记忆元Ctemp
Ctemp_t = tanh(X_t@W_xc + H_t−1@W_hc + b_c)
记忆元C，⊙为按元素乘法
C_t = F_t⊙C_t−1 + I_t⊙Ctemp_t
隐状态输出公式
H_t = O_t⊙tanh(C_t).
'''


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal(shape=(num_inputs, num_hiddens)),
                normal(shape=(num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 初始化参数
    W_xf, W_hf, b_f = three()  # 遗忘门F的参数
    W_xi, W_hi, b_i = three()  # 输入门I的参数
    W_xo, W_ho, b_o = three()  # 输出门O的参数
    W_xc, W_hc, b_c = three()  # 候选记忆元Ctemp的参数
    # 输出层参数(注意与输出门区分，输出门是计算记忆元的，而输出层是输出最后结果的)
    W_hq = normal(shape=(num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)
    return params


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i,
     W_xf, W_hf, b_f,
     W_xo, W_ho, b_o,
     W_xc, W_hc, b_c,
     W_hq, b_q] = params
    # 初始状态包含H0和C0
    (H, C) = state
    outputs = []
    for X in inputs:
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        Ctemp = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        # 记忆元更新公式
        C = F * C + I * Ctemp
        # 隐状态更新公式
        H = O * torch.tanh(C)
        # 该层输出
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


def init_lstm_state(batch_size, num_hiddens, device):
    # 在lstm中，初始化状态需要两个量，即H0和初始的C0
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def train_diy_lstm():
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                                init_lstm_state, lstm)
    demo8_5.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    # 从零实现
    # train_diy_lstm()
    # PyTorchAPI实现
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model.to(device)
    demo8_5.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
'''PyTorchAPI实现
困惑度 1.0, 249342.9 词元/秒 cuda:0
time travelleryou can show black is white by argument said filby
travelleryou can show black is white by argument said filby
'''