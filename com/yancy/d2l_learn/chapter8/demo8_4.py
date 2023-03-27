import torch
from d2l import torch as d2l

'''
循环神经网络RNN
前向公式含有隐状态H_(t-1)及其权重W_hh，如果去掉H_(t-1)@W_hh就退化成MLP
隐变量更新: H_t = phi(X_t@W_xh + H_(t-1)@W_hh + b_h)
从隐变量输出: O_t = phi(H_t@W_hq + b_q)

困惑度=exp(平均交叉熵): 1最好，无穷大最差
梯度裁剪:g=max(1,theta/g.norm())*g 即如果g大于theta，则将g重置(g和g.norm约掉，保留符号)为theta，否则不变(1*g)
'''

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
