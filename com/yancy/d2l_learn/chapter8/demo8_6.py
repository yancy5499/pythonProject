import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter8.demo8_5 as demo_8_5

'''
RNN简洁实现
'''


class RNNModel(nn.Module):
    """循环神经⽹络模型"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    # rnn_layer是隐藏状态层
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    # H0初始化(隐藏层数，batch_size,隐藏单元数)
    # 当多层隐藏层时，前一层的输出作为下一层的输入
    state = torch.zeros((1, batch_size, num_hiddens))
    # 独热编码后的标准化输入X=one_hot(X0.T,vocab_size), size=(num_step,batch_size,vocab_size)
    X = torch.rand(num_steps, batch_size, len(vocab))
    Y, state_new = rnn_layer(X, state)
    # torch.Size([num_step, batch_size, num_hiddens]) torch.Size([1, batch_size, num_hiddens])
    print(Y.shape, state_new.shape)

    device = d2l.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net.to(device)
    print('训练前:')
    print(d2l.predict_ch8('time traveller', 10, net, vocab, device))
    num_epochs, lr = 500, 1
    print('训练中...')
    demo_8_5.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
'''
困惑度 1.3, 359203.7 词元/秒 cuda:0
time traveller of ceedit yomare trove thing ce ton said ththe an
traveller with a ulight accessallige thanicand scay o jered
'''
