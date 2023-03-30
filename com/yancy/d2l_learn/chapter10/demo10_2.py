import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
注意力汇聚: Nadaraya-Watson核回归
'''


def f(x):
    return 2 * torch.sin(x) + x ** 0.8


def plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat):
    plt.figure(figsize=(5, 4))
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.show()


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 可学习的权重
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, q, k, v):
        q = q.repeat_interleave(k.shape[1]).reshape(-1, k.shape[1])
        # query与keys做距离运算得到注意力权重
        self.attention_weights = nn.functional.softmax(
            -((q - k) * self.w) ** 2 / 2, dim=1
        )
        # 注意力权重与values按batch矩阵乘法得到输出
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         v.unsqueeze(-1)).reshape(-1)


if __name__ == '__main__':
    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 排序
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 加噪音
    x_test = torch.arange(0, 5, 0.1)
    y_truth = f(x_test)

    # 开始估计，选择平均汇聚
    y_hat = torch.repeat_interleave(y_train.mean(), len(x_test))
    plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat)

    # =====非参数Nadaraya-Watson核回归=====
    # repeat_interleave不加参数时，将x_test展平后，每个元素重复n_train次，然后reshape
    X_repeat = x_test.repeat_interleave(n_train).reshape(-1, n_train)
    # 相当于将一维x_test升维成二维列向量后，将列重复n_train次
    print(x_test.unsqueeze(dim=1).repeat_interleave(n_train, dim=1) == X_repeat)

    # X_repeat - x_train相当于将两个一维的张量x_test和x_train两两元素之间的距离，存储为一个二维矩阵
    # 比如a=[1,2,3],b=[1,1,1]
    # a_repeat=[[1,1,1],[2,2,2],[3,3,3]],和广播机制后的b进行相减
    # 得到[[0,0,0],[1,1,1],[2,2,2]]，该张量存储了ai与bj之间的距离信息，矩阵的(i,j)位置即ai与bj的距离
    # 相当于a的第一个元素与b的所有元素距离做第一列，第二个元素与b所有元素距离做第二列，依次类推
    attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat)
    d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs',
                      figsize=(4, 4))
    plt.show()

    # =====学习参数的NW核回归=====
    # 批量矩阵乘法torch.bmm(X_batch,Y_batch)，即忽视第一个batch维度，从X和Y中按batch取出对应的元素，进行矩阵乘法，再按batch存储起来
    X = torch.ones(3, 3).type(torch.float32)
    Y = torch.arange(9).reshape(3, -1).type(torch.float32)
    X_batch = torch.stack([torch.zeros_like(X), X])
    Y_batch = torch.stack([Y, Y])
    print(torch.bmm(X_batch, Y_batch))
    # 按0轴复制n_train次，作为batch
    X_tile = x_train.repeat((n_train, 1))
    Y_tile = y_train.repeat((n_train, 1))
    # 将对角线归0，表示做键值对计算时，只关心除自己以外的其他数据的距离
    # 因此先用(1 - torch.eye(n_train)).type(torch.bool)做了一个对角线为false的mask
    # 用于X_tile[mask]消除掉对角线元素，输出展平的一维张量，再reshape，此时大小已经变成了(n_train, n_train-1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # print(X_tile.shape,keys.shape)

    # 训练
    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    # 估计
    queries = x_test
    keys = x_train.repeat((len(x_test), 1))
    values = y_train.repeat((len(x_test), 1))
    y_hat = net(queries, keys, values).unsqueeze(1).detach()
    plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat)
    d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs',
                      figsize=(4, 4))
    plt.show()
