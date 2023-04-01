import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter9.demo9_5 as demo9_5
import com.yancy.d2l_learn.chapter9.demo9_7 as demo9_7
import matplotlib.pyplot as plt

'''
transformer
'''


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # 进来的X分别通过多头自注意力，和恒等映射，最后通过addNorm相加
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # 再做一次
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            # 重复num_layers次的EncoderBlock
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size,
                             num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias)
            )

    def forward(self, X, valid_lens, *args):
        # 正余弦位置编码处于-1到1之间
        # 嵌入值乘以嵌入维度的平方根进行缩放
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        # 输出(batch_size, num_step, num_hiddens)
        return X


class DecoderBlock(nn.Module):
    """解码器块(每一个都需要接受编码器的输出)"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            # 训练阶段, state[2][self.i]会初始化为None
            key_values = X
        else:
            # 预测阶段, state[2][self.i]包含着前文信息(解码器输出)
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每⼀行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 训练阶段X=key_values，即进行自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器-解码器注意力(编码器输出作为kv)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i)
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 用该属性存储两种注意力层的权重（一个是自注意力，一个是编码-解码注意力）
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 存储权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


def test():
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    print(encoder_blk(X, valid_lens).shape)

    print('=' * 10)


if __name__ == '__main__':
    # test()
    # ==========
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    # 读取数据集
    train_iter, src_vocab, tgt_vocab = demo9_5.load_data_nmt(batch_size, num_steps)
    # 定义网络架构
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    # 训练
    demo9_7.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    # 预测并检验
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ==>',
              f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
    # ==========
    # 读取注意力权重，进行可视化
    # 编码器注意力可视化
    enc_attention_weights = torch.cat(
        net.encoder.attention_weights, 0
    ).reshape((num_layers, num_heads, -1, num_steps))
    d2l.show_heatmaps(
        enc_attention_weights.cpu(), xlabel='Key positions',
        ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
        figsize=(7, 3.5))
    plt.show()
    # 解码器注意力可视化
    dec_attention_weights_2d = [head[0].tolist()
                                for step in dec_attention_weight_seq  # 从输出的注意力权重属性中取step
                                for attn in step  # 再从step中取attn
                                for blk in attn  # 再从attn中取blk
                                for head in blk]  # 再从blk里取head
    dec_attention_weights_filled = torch.tensor(
        pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
    dec_attention_weights = dec_attention_weights_filled.reshape(
        (-1, 2, num_layers, num_heads, num_steps))
    # 取出解码器的两种注意力权重
    dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)
    # 解码器自注意力权重
    d2l.show_heatmaps(
        dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
        xlabel='Key positions', ylabel='Query positions',
        titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
    plt.show()
    # 解码器的编码解码注意力权重
    d2l.show_heatmaps(
        dec_inter_attention_weights, xlabel='Key positions',
        ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
        figsize=(7, 3.5))
    plt.show()
"""
loss 0.032, 9991.4 tokens/sec on cuda:0
go . => va !, ==> bleu 1.000
i lost . => j'ai perdu ., ==> bleu 1.000
he's calm . => il est calmes ., ==> bleu 0.658
i'm home . => je suis chez moi ., ==> bleu 1.000
"""