import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
自注意力和位置编码
'''


def test_SelfAttention():
    print('=====>test_SelfAttention')
    num_hiddens, num_heads = 100, 5
    attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                       num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    # 不同于10-5的XYY，此处的XXX即X同时做qkv，即自注意力
    print(attention(X, X, X, valid_lens).shape)
    print('=' * 10)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 存储位置编码
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # 基于正余弦函数的固定位置编码
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def test_PositionalEncoding():
    print('=====>test_SelfAttention')
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    # 行代表词元在序列中的位置，列代表位置编码不同的维度
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
             figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
    plt.show()
    P = P[0, :, :].unsqueeze(0).unsqueeze(0)
    d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                      ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
    plt.show()
    print('=' * 10)


if __name__ == '__main__':
    test_SelfAttention()
    test_PositionalEncoding()
