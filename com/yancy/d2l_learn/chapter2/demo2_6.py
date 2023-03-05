import torch
from torch.distributions import multinomial
import matplotlib.pyplot as plt

'''
概率分布相关
'''
# 概率向量
pxi = torch.ones([6]) / 6  # [1/6,1/6,1/6,1/6,1/6,1/6]
# 生成分布,1次实验
P = multinomial.Multinomial(1, pxi)
# 生成一个实验样本
print(P.sample())
# 重复100组的实验
P100 = multinomial.Multinomial(100, pxi)
# 生成5个样本，每个样本做一次实验，一次实验包含500组重复
print(P100.sample((5,)))
# 验算频率
print(P100.sample() / 100)
P10000 = multinomial.Multinomial(10000, pxi)
print(P10000.sample() / 10000)
print('=' * 10)
# 绘图展示概率收敛
samples_tensor = multinomial.Multinomial(10, pxi).sample((1000,))
# 每行为一组样本数据，按行不断往下累加
# 即第一行为第一个样本的数据，第二行为一二行合计的数据，行数越多频率越收敛于概率
cum_samples_tensor = samples_tensor.cumsum(axis=0)
# 归一化后为频率tensor
freq_tensor = cum_samples_tensor / cum_samples_tensor.sum(axis=1, keepdims=True)
# print(freq_tensor)
# print(cum_samples_tensor.sum(axis=1, keepdims=True))
# 开始画图
plt.figure(figsize=(10, 10), dpi=128)
times = torch.linspace(-10, 1000, steps=1000)
# print(times)
plt.xlabel('times')
plt.ylabel('freq')
# 频率收敛线，过(0,1/6)点，斜率为0
plt.axline((0, 1 / 6), slope=0, label='freq=1/6', color='black', linestyle='dashed')
for i in range(6):
    # 第i面筛子的频率收敛曲线,y值为freq_tensor的第i列
    plt.plot(times, freq_tensor[:, i].numpy(),
             label='P(point={})'.format(i),
             linewidth=1)
plt.legend(loc='upper right')
plt.show()
