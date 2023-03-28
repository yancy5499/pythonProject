import torch
import os
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
机器翻译与数据集(简单了解)
'''


def read_data_nmt():
    """载入'英语-法语'数据集"""
    if not os.path.exists('../data/fra-eng'):
        data_dir = d2l.download_extract('fra-eng')
        os.remove(os.path.join('../data', 'fra-eng.zip'))
    else:
        data_dir = '../data/fra-eng'
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
              encoding='utf-8') as f:
        return f.read()


def load_data_nmt(batch_size, num_steps, num_examples=600):
    text = d2l.preprocess_nmt(read_data_nmt())
    source, target = d2l.tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = d2l.build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = d2l.build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


if __name__ == '__main__':
    raw_text = read_data_nmt()
    # print(raw_text[:75])

    # 文本预处理
    # 包括：使用空额替换不间断空格，使用小写字母替换大写字母，在单词和标点符号之间插入空格
    text = d2l.preprocess_nmt(raw_text)
    # print(text[:80])

    # 词元化（单词、标点符号）
    source, target = d2l.tokenize_nmt(text)
    # print(source[:6], '\n', target[:6])

    fig = plt.figure(figsize=(5, 5), dpi=200)
    d2l.show_list_len_pair_hist(['source', 'target'],
                                '# tokens per sequence',
                                'count',
                                source, target)
    plt.show()

    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])  # 特定词元：填充、开始、结束
    # 填充与截断:文本序列词元数小于num_step填充，反之截断
    print(
        d2l.truncate_pad(src_vocab[source[0]],
                         num_steps=10,
                         padding_token=src_vocab['<pad>'])  # 用于填充的词元
    )
    # 文本序列转换成小批量数据集: d2l.build_array_nmt(lines, vocab, num_steps) >>return array, valid_len
    # 序列的结尾会添加<eos>, 记录了文本序列的长度(排除了末尾的<pad>)

    # 数据迭代器，同时还会返回两种词表
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效⻓度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效⻓度:', Y_valid_len)
        break
