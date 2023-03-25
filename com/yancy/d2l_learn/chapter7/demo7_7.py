import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter6.demo6_6 as demo6_6

'''
DenseNet稠密连接网络
'''


def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 通道维度上连接
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


if __name__ == '__main__':
    blk = DenseBlock(2, 10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)
    transblk = transition_block(10)  # 通过过渡层将其通道数降为10
    print(transblk(Y).shape)
    print('=' * 10)

    b1 = nn.Sequential(
        nn.LazyConv2d(64, kernel_size=7, stride=2, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_block = [4, 4, 4, 4]
    blks = []
    for i, num_covs in enumerate(num_convs_in_dense_block):
        blks.append(DenseBlock(num_covs, growth_rate))
        num_channels += num_covs * growth_rate
        if i != len(num_convs_in_dense_block) - 1:
            # 添加过渡层，使通道减半
            num_channels = num_channels // 2
            blks.append(transition_block(num_channels))
    net = nn.Sequential(b1, *blks,
                        nn.LazyBatchNorm2d(),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.LazyLinear(10))
    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    demo6_6.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.128, train acc 0.953, test acc 0.875
680.5 examples/sec on cuda:0
'''