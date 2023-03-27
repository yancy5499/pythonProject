import random
import torch
from d2l import torch as d2l

'''
语言模型
'''
# 数据元迭代器，随机采样(起始偏移随机)连续的num_step子序列，序列整体后移一位作为Y，一次取batch_size个数据元组合
# d2l.seq_data_iter_random(corpus, batch_size, num_steps)
my_seq = list(range(35))
for X, Y in d2l.seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
print('=' * 10)
# 序列采样，与随机不同，该采样保证了连续两个采样是连续的
# d2l.seq_data_iter_sequential(corpus, batch_size, num_steps)
for X, Y in d2l.seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
# 整合
# d2l.SeqDataLoader(batch_size,
#                   num_steps,
#                   use_random_iter,
#                   max_tokens)
