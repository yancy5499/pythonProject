import collections
import re
from d2l import torch as d2l

'''
文本预处理
'''


def init_data():
    lines = d2l.read_time_machine()
    print(f'# ⽂本总⾏数: {len(lines)}')
    print(lines[0])
    print(lines[10])
    print('数据准备完毕！')
    return lines


# 整合所有步骤
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = d2l.read_time_machine()
    tokens = d2l.tokenize(lines, 'char')  # 注意用的char，即词元是单个字母
    vocab = d2l.Vocab(tokens)
    # 因为时光机器数据集中的每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，
    # 所以将所有⽂本⾏展平到⼀个列表中
    corpus = [vocab[token] for one_line_token in tokens for token in one_line_token]
    # 连续for从左往右生效,先取到tokens中的one_line_token，再在one_line_token里取token给最前面的用
    # vocab[token]即取token的索引，将文本数据转化为数字数据
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':
    lines = init_data()
    # 将每行拆分成单词元列表lines[0].split(),无参数的split默认用空格分割
    tokens = d2l.tokenize(lines)
    for i in range(11):
        print(tokens[i])
    # 生成文本词表(将数据分成词元后统计每个词出现的频率，然后根据频率大小赋予数字索引，以便于模型计算)
    # 特殊的未知词元<unk>，填充词元<pad>，序列开始<bos>，序列结束<eos>
    vocab = d2l.Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])
    print(vocab.__getitem__('the') == vocab['the'])  # 等价操作

    print(load_corpus_time_machine(10)[1].token_to_idx.items())
