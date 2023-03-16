import torch

'''
数值稳定性与模型初始化
'''
# 初始化不佳可能导致数据溢出
M = torch.normal(0, 1, size=(4, 4))
print(M)
for i in range(100):
    # M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))
    M = M @ torch.normal(0, 1, size=(4, 4))
print(M)

# 默认初始化,不指定初始化方式时框架将使用默认的随机初始化方法，对于中等难度问题有效

# Xavier初始化,同时考虑前向传播与方向传播的方差变化，设定系数使两个方向的方差都不会变得太大
# 0.5*(n_in+n_out)sigma^2或sigma^2=2/(n_in+n_out)
