import torch
from torch import nn

'''
卷积相关知识
'''


def corr2d(X, K):
    """计算二维互相关"""
    h, w = K.shape
    # 输出大小为(nh-kh+1)*(nw-kw+1),可从运算时的滑动情况得知，若输入3*3，核2*2，
    # 则在运算时，核可在输入中滑动一次，一行有两次输出(1+1)
    # 若有padding和stride，则输出大小变为[(nh+ph-kh+1)//sh]*[(nw+pw-kw+1)//sw] # 用地板除
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# tips:padding用于减少卷积时的图像边缘损失
#      stride用于减少图像的高分辨率带来的计算压力、加快计算速度等

def corr2d_multi_in(X, K):  # 仍然叫二维卷积，只是因为二维图像的三通道分为三次计算累积而已，真正的三维卷积是时间维度的
    h, w = K[0].shape
    Y = torch.zeros((X.shape[1] - h + 1, X.shape[2] - w + 1))
    # 根据0轴深度遍历计算2d
    for i in range(X.shape[0]):
        Y = Y[:] + corr2d(X[i], K[i])
    # 简洁实现如下:
    # Y = sum(corr2d(x, k) for x, k in zip(X, K))
    return Y


def corr2d_multi_in_out(X, K):  # 输出也变成多通道，不再累加，而是堆叠
    # [corr2d(x, k) for x, k in zip(X, K)]运算后会返回一个列表
    # 用torch.stack()将列表中的元素以一个新的轴堆叠起来，同时该新轴作为0轴
    return torch.stack([corr2d(x, k) for x, k in zip(X, K)], 0)


# pooling层，用于汇聚数据，与卷积类似，但是核参数是固定的
# pooling多通道时默认输出也是多通道，不累加
def pool2d(X, pool_size, mode='max'):
    # 简化的pool2d = nn.MaxPool2d(pool_size,padding,stride)
    # 其中nn.MaxPool2d的size为一个数时默认为方形，默认stride=pool_size
    h, w = pool_size
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mode == 'max':
                Y[i, j] = (X[i:i + h, j:j + w]).max()
            elif mode == 'avg':
                Y[i, j] = (X[i:i + h, j:j + w]).mean()  # 若令h*w=n，相当于与元素为1/n的核做卷积
    return Y


# 用于理解卷积网络的前向传播
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias


if __name__ == '__main__':
    # X = torch.arange(9).reshape(3, -1)
    # K = torch.arange(4).reshape(2, -1)
    # print(corr2d(X, K))
    X = torch.ones(6, 8)
    X[:, 2:6] = 0  # 令第三列到第六列为0
    print(X)
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    print(corr2d(X.T, K))
    print('=' * 10)
    # 输入1输出1
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2
    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # 更新卷积核
        conv2d.weight.data = conv2d.weight.data[:] - lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print('epoch {}, loss {:.3f}'.format(i + 1, l.sum()))
    print(conv2d.weight.data.reshape(1, 2))
    print('=' * 10)
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]],
                      [[1.0, 2.0], [3.0, 4.0]]])
    print(corr2d_multi_in(X, K))
    print(corr2d_multi_in_out(X, K))
