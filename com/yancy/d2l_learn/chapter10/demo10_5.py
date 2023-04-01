import torch
from torch import nn
from d2l import torch as d2l

'''
多头注意力
'''


def transpose_qkv(X, num_heads):
    """为了多头注意力更改shape"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    # 输出(batch_size*num_heads, num_qkv, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    # 转换output形状，逆转transpose_qkv的影响
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        # 缩放点积注意力
        self.attention = d2l.DotProductAttention(dropout)
        # 初始化权重
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, q, k, v, valid_lens):
        # q: (batch_size, num_q, num_hiddens)
        # k,v: (batch_size, num_kv, num_hiddens)
        # valid_lens: (batch_size,) or (batch_size, num_q)
        # 变换后: (batch_size*num_heads, num_qkv, num_hiddens/num_heads)
        q = transpose_qkv(self.W_q(q), self.num_heads)
        k = transpose_qkv(self.W_k(k), self.num_heads)
        v = transpose_qkv(self.W_v(v), self.num_heads)
        if valid_lens is not None:
            # 在0轴将元素复制num_heads次
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # output: (batch_size*num_heads, num_q, num_hiddens/num_heads)
        output = self.attention(q, k, v, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def test_MultiHeadAttention():
    print('=====test MultiHeadAttention=====')
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
    attention.eval()
    print(attention)
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)
    print('=' * 30)


if __name__ == '__main__':
    test_MultiHeadAttention()
