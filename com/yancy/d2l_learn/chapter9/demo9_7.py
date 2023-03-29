import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
from com.yancy.d2l_learn.chapter3.demo3_7 import MyPlot
import matplotlib.pyplot as plt
import com.yancy.d2l_learn.chapter9.demo9_5 as demo9_5

'''
seq2seq学习
<bos><eos>用于判断序列的起止
编码器的最终隐状态，用于初始化解码器的隐状态
'''


class Seq2SeqEncoder(d2l.Encoder):
    """编写具体的Seq2Seq编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
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

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # =====原代码=====
        self.rnn = nn.GRU(embed_size + num_hiddens,
                          num_hiddens,
                          num_layers,
                          dropout=dropout)
        # =====原代码=====
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # enc_outputs=[output, state]
        return enc_outputs[1]

    def forward(self, X, state):
        # 直接得到标准维度的X, (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # =====原代码=====
        # 增广state，使其第一个维度重复X.shape[0]次，其余两个维度不变
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 按2轴连接X_context
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        # =====原代码=====
        output = self.dense(output).permute(1, 0, 2)
        # output shape: (batch_size,num_steps,vocab_size)
        # state shape: (num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder_withoutContext(d2l.Decoder):
    """编写具体的Seq2Seq解码器(修改)"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqDecoder_withoutContext, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # =====修改=====
        # 不连接X和context后，输入维度也需要跟着修改
        self.rnn = nn.GRU(embed_size,
                          num_hiddens,
                          num_layers,
                          dropout=dropout)
        # =====修改=====
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # enc_outputs=[output, state]
        return enc_outputs[1]

    def forward(self, X, state):
        # 直接得到标准维度的X, (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # =====修改=====
        # 不再连接X和context，因为编码器的信息已经包含在了state里，不再需要自己取出context
        output, state = self.rnn(X, state)
        # =====修改=====
        output = self.dense(output).permute(1, 0, 2)
        # output shape: (batch_size,num_steps,vocab_size)
        # state shape: (num_layers,batch_size,num_hiddens)
        return output, state


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    # 先生成一个最大长度的一维张量maxlen_tensor=[0,1,2,...,maxlen-1]
    # 然后与valid_len比较，valid_len是一个一维张量，例如[1,3,2]
    # [None,:]可理解为将maxlen_tensor复制多行，[:,None]将valid_len转置后复制多列
    # 然后二者进行逻辑<比较，显然第一行[0,1,2,...]和[1,1,1,...]比较，只有第一个True
    # 后者同上，最后的结果就是，复制valid_len倍后的maxlen_tensor，每行True的长度就是valid_len对应列的数字大小
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # ~mask即对mask按位取反(1to0, 0to1)，并作为索引，赋值value
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    def forward(self, pred, label, valid_len):
        # 先生成全1张量，形状与label相同
        weights = torch.ones_like(label)
        # 通过mask，用我们已知的各句子的长度valid_len，把weights张量更新为真实长度的部分为1，其余为0
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # 用weights更新损失，只保留有效的部分
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    # =====画图=====
    fig = plt.figure()
    x_values = torch.linspace(10, num_epochs, num_epochs // 10)
    myplot = MyPlot(fig, x_values)
    # =====画图=====
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 记录训练损失综合、词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor(
                [tgt_vocab['<bos>']] * Y.shape[0], device=device
            ).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 拼上bos，强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            myplot.add_y(metric[0] / metric[1])
        print('\repoch {:.1f}%'.format(100 * (epoch + 1) / num_epochs), end='')
    print(f'\nloss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
    myplot.show()


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    # 预测时将net设置为评估模式(会让dropout失效)
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split()] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device),
        dim=0
    )
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device),
        dim=0
    )
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用最高可能性的词元，作为解码器在下一个时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 检测序列结束
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """预测结果的评估指标"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


if __name__ == '__main__':
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()  # 纯计算模式
    X = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(X)
    print(output.shape, state.shape)

    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()  # 纯计算模式
    # 由编码器的最后输出隐状态初始化解码器的隐状态(输入的encoder(X)有两个输出, init_state会自动取出state)
    state = decoder.init_state(encoder(X))
    output, state = decoder(X, state)
    print(output.shape, state.shape)

    X = torch.ones(2, 3, 4)
    # torch.tensor([1, 2])意味着第一个维度只保留1，第二个维度只保留2，其余用value替代
    print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

    loss = MaskedSoftmaxCELoss()
    print(loss(torch.ones(3, 4, 10),
               torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0])))
    print('=' * 10)

    # 训练
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = demo9_5.load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab),
                             embed_size,
                             num_hiddens,
                             num_layers,
                             dropout)
    # 此处可选用Seq2SeqDecoder_withoutContext
    decoder = Seq2SeqDecoder(len(tgt_vocab),
                             embed_size,
                             num_hiddens,
                             num_layers,
                             dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 预测
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    # 标准答案
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    print('预测结果:')
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device
        )
        print(f'{eng} => {translation} ==> bleu {bleu(translation, fra, k=2):.3f}')

'''
loss 0.019, 34432.9 tokens/sec on cuda:0
预测结果:
go . => va ! ==> bleu 1.000
i lost . => j'ai perdu . perdu . ==> bleu 0.651
he's calm . => il est paresseux fais tomber fais ? ==> bleu 0.342
i'm home . => je suis détendu ici . ==> bleu 0.548
'''
'''
# =====修改后=====
loss 0.020, 31858.9 tokens/sec on cuda:0
预测结果:
go . => va ! ==> bleu 1.000
i lost . => j'ai perdu . ==> bleu 1.000
he's calm . => il est mouillé . ==> bleu 0.658
i'm home . => je suis chez moi . ==> bleu 1.000
'''
