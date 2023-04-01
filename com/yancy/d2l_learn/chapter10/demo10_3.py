import torch
import math
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
注意力评分函数
attention_weights = softmax(a), 其中a(q,k)是评分函数
用于衡量qk之间的距离
'''


def masked_softmax(X, valid_lens):
    """添加遮蔽的softmax"""
    # X:3D张量，valid_lens:1D or 2D张量
    if valid_lens is None:
        # dim=-1按最后一个轴加和为1，做softmax操作
        return nn.functional.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        # repeat_interleave未指定轴，展平然后每个元素重复shape[1]次
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        # 展平
        valid_lens = valid_lens.reshape(-1)
    # 最后一个轴上被掩蔽的元素使用一个非常大的复制替换，使其softmax输出为0
    # 如三维时，最后一个轴为列，mask根据valid_lens的值从列方向阶段X，
    # 使其超过valid_lens的列为value，该value通过softmax后会变成0
    # print(X)
    # print(X.reshape(-1, shape[-1]))
    X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力（不要求qk同size）"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        # W_k和W_q负责将qk的不同数量，结果一个线性层统一抽象为num_hiddens
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # 通过w_v后只输出一个值
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, valid_lens):
        q, k = self.W_q(q), self.W_k(k)
        # 为了广播计算后控制feature的形状
        # q升维成(batch_size, num_q, 1, num_hiddens)
        # k升维成(batch_size, 1, num_kv, num_hiddens)
        features = q.unsqueeze(2) + k.unsqueeze(1)
        # features的形状为(batch_size, num_q, num_kv, num_hiddens)
        # print(features.shape)
        # features = torch.tanh(features)
        # 消除最后一个维度
        # scores形状: (batch_size, num_q, num_kv)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # v的形状: (batch_size, num_kv, dim_v)
        # 输出: (batch_size, 查询步数, 值的维度)
        return torch.bmm(self.dropout(self.attention_weights), v)


class DotProductAttention(nn.Module):
    """缩放点积注意力(要求qk同size=d)"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, valid_lens=None):
        d = q.shape[-1]
        # tensor.transpose(1, 2)忽略0轴，当成12轴组成的二维张量进行转置
        # 对于二维tensor, tensor.T==tensor.transpose(0,1)
        # 对于三维tensor, tensor.T==tensor.transpose(0,2)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), v)


if __name__ == '__main__':
    # 输入的三维X会先消除0轴即batch_size维度，将其按1轴拼接在一起
    # 输入的valid_lens是一维张量，会进行重复元素，按顺序对X每行的元素数量进行限制
    # 最后的效果就是对两个批次的len限制为2和3
    print(masked_softmax(torch.rand(2, 2, 4),
                         valid_lens=torch.tensor([2, 3])))
    # 输入二维张量的valid_lens，会展平
    # 然后将消除batch_size维度的X按顺序进行元素限制
    # 最后的效果是bacth1限制长度为[2,3], batch2限制长度为[3,1]
    print(masked_softmax(torch.rand(2, 2, 4),
                         valid_lens=torch.tensor([[2, 3], [3, 1]])))
    print('=' * 10)
    # =====加性注意力(qk可不同长度)=====
    # 评分函数a(q,k) = w_v.T@tanh(W_q@q + W_k@k)
    # num_q=1, query_size=20
    # num_k=10, key_size=2, 其中num_qkv可解释为步数或词元序列长度
    queries = torch.normal(0, 1, (2, 1, 20))
    keys = torch.ones((2, 10, 2))
    # 修改keys值
    # keys[:, :3, 0] = 0
    # values形状(2,10,4), num_k=num_v, 在0轴处重复了两次
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                  dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))
    # 由于本例中每个键都是相同的1，所有注意力权重是均匀的，只有有效长度决定（因为非有效长度被mask了）
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries',
                      figsize=(4, 4))
    plt.show()
    print('=' * 10)
    # =====缩放点积注意力(qk同size)=====
    # 评分函数a(q,k) = q.T@k/sqrt(d)
    queries_qsize_equal_ksize = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    attention(queries_qsize_equal_ksize, keys, values, valid_lens)
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries',
                      figsize=(4, 4))
    plt.show()
