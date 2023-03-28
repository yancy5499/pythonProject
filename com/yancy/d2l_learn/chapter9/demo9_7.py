import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

'''
seq2seq学习
<bos><eos>用于判断序列的起止
编码器的最终隐状态，用于初始化解码器的隐状态
'''


class Seq2SeqEncoder(d2l.Encoder):
    """编写具体的Seq2Seq编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输入shape=(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # 在rnn中，第一个轴对应于时间步, 用permute函数改变维度:(0,1,2)>>(1,0,2)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        # output shape: (num_steps,batch_size,num_hiddens)
        # state shape: (num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder(d2l.Decoder):
    """编写具体的Seq2Seq解码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens,
                          num_hiddens,
                          num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 直接得到标准维度的X, (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 增广state，使其第一个维度重复X.shape[0]次，其余两个维度不变
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 按2轴连接X_context
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output shape: (batch_size,num_steps,vocab_size)
        # state shape: (num_layers,batch_size,num_hiddens)
        return output, state


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵巡视函数"""

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


if __name__ == '__main__':
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()  # 纯计算模式
    X = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(X)
    print(output.shape, state.shape)

    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()  # 纯计算模式
    # 由编码器的最后输出隐状态初始化解码器的隐状态
    state = decoder.init_state(encoder(X))
    output, state = decoder(X, state)
    print(output.shape, state.shape)

    X = torch.ones(2, 3, 4)
    # torch.tensor([1, 2])意味着第一个维度只保留1，第二个维度只保留2，其余用value替代
    print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

    loss = MaskedSoftmaxCELoss()
    print(loss(torch.ones(3, 4, 10),
               torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0])))

    # 训练
    # ToDo
