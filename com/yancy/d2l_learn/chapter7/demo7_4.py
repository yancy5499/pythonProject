import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter6.demo6_6 as demo6_6

'''
GoogLeNet
'''


class Inception(nn.Module):
    # c1到c2是四条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))  # MaxPool2d不用接relu
        return torch.cat((p1, p2, p3, p4), dim=1)


class LazyInception(nn.Module):
    # c1到c2是四条路径的输出通道数
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(LazyInception, self).__init__(**kwargs)
        self.p1_1 = nn.LazyConv2d(c1, kernel_size=1)

        self.p2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.p2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.p3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))  # MaxPool2d不用接relu
        return torch.cat((p1, p2, p3, p4), dim=1)


def testLazyInitNet():
    b1 = nn.Sequential(nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(nn.LazyConv2d(64, kernel_size=1),
                       nn.ReLU(),
                       nn.LazyConv2d(192, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(LazyInception(64, (96, 128), (16, 32), 32),
                       LazyInception(128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b4 = nn.Sequential(LazyInception(192, (96, 208), (16, 48), 64),
                       LazyInception(160, (112, 224), (24, 64), 64),
                       LazyInception(128, (128, 256), (24, 64), 64),
                       LazyInception(112, (144, 288), (32, 64), 64),
                       LazyInception(256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b5 = nn.Sequential(LazyInception(256, (160, 320), (32, 128), 128),
                       LazyInception(384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten())
    return nn.Sequential(b1, b2, b3, b4, b5, nn.LazyLinear(10))


if __name__ == '__main__':
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    X = torch.rand(size=(1, 1, 96, 96))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    print('=' * 10)
    lazy_net = testLazyInitNet()
    X = torch.rand(size=(1, 1, 96, 96))
    for layer in lazy_net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    # lr, num_epochs, batch_size = 0.1, 10, 128
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    # demo6_6.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
