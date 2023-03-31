import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter9.demo9_5 as demo9_5
import com.yancy.d2l_learn.chapter9.demo9_7 as demo9_7
import matplotlib.pyplot as plt

'''
使用Bahdanau注意力的Seq2Seq
'''


class AttentionDecoder(d2l.Decoder):
    """带有注意力机制的编码器基本接口"""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            key_size=num_hiddens, query_size=num_hiddens, num_hiddens=num_hiddens, dropout=dropout
        )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens,
                          num_hiddens,
                          num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    @property
    def attention_weights(self):
        return self._attention_weights

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # state中包含编码器的输出，隐状态，和编码器过来的有效长度张量，由编码器的输出初始化而来
        # 其中enc_outputs=kv, hidden_state=q计算注意力，之后注意力和解码器的输入连接
        # 连接后的张量作为输入，hidden_state作为隐状态，输入rnn
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # 用q=hidden_state[-1],k=v=enc_outputs做注意力计算
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 将注意力作为上下文和x连接后一起输入rnn
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 计算输出，并更新隐状态
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(tensors=outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]


class Seq2SeqEncoder_lstm(d2l.Seq2SeqEncoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0, **kwargs):
        # 先用父类初始化所有属性，但是rnn不是我们想要的
        super().__init__(vocab_size, embed_size, num_hiddens, num_layers,
                         dropout, **kwargs)
        # 重写rnn，使用LSTM
        self.rnn = nn.LSTM(
            embed_size,
            num_hiddens,
            num_layers=num_layers,
            dropout=dropout
        )


class Seq2SeqAttentionDecoder_lstm(Seq2SeqAttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0, **kwargs):
        super().__init__(vocab_size, embed_size, num_hiddens, num_layers,
                         dropout, **kwargs)
        self.rnn = nn.LSTM(
            embed_size + num_hiddens,
            num_hiddens,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # 从lstm输出的hidden_state=(H,C)，取H作为键
            query = torch.unsqueeze(hidden_state[0][-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 将注意力作为上下文和x连接后一起输入rnn
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 计算输出，并更新隐状态
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(tensors=outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]


def test_Decoder():
    encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                 num_layers=2)
    encoder.eval()
    decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                      num_layers=2)
    decoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    print('=====test Decoder=====')
    print('\n'.join(
        [str(i) for i in [output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape]]
    ))
    print('=' * 10)


if __name__ == '__main__':
    test_Decoder()
    # 训练
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = demo9_5.load_data_nmt(batch_size, num_steps)
    # encoder = d2l.Seq2SeqEncoder(
    #     len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    # decoder = Seq2SeqAttentionDecoder(
    #     len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    # =====lstm替换=====
    encoder = Seq2SeqEncoder_lstm(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder_lstm(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    # ============
    net = d2l.EncoderDecoder(encoder, decoder)
    demo9_7.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    # 测试
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ==>',
              f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
    # 查看注意力权重
    attention_weights = torch.cat(
        [step[0][0][0] for step in dec_attention_weight_seq], 0
    ).reshape((1, 1, -1, num_steps))
    d2l.show_heatmaps(
        attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
        xlabel='Key positions', ylabel='Query positions',
        figsize=(4, 6)
    )
    plt.show()
'''
loss 0.021, 10921.1 tokens/sec on cuda:0
go . => va !, ==> bleu 1.000
i lost . => j'ai perdu ., ==> bleu 1.000
he's calm . => il est bon ., ==> bleu 0.658
i'm home . => je suis chez moi ., ==> bleu 1.000
'''
