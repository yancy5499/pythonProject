import time
import torch

'''
矢量化加速
'''

n = 10000
a = torch.ones([n])
b = torch.ones([n])
c = torch.zeros([n])
start_time = time.time()
# for循环法
for i in range(n):
    c[i] = a[i] + b[i]
end_time = time.time()
print('for用时{}s'.format(end_time - start_time))
# ===============
start_time = time.time()
# 矢量化运算
d = a + b
end_time = time.time()
print('矢量用时{}s'.format(end_time - start_time))
