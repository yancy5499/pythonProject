import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter6.demo6_6 as demo6_6

'''
VGG网络
'''


def vgg_block(num_convs, in_channels, out_channnels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channnels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channnels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blks,
                         # MLP部分
                         nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),

                         nn.Linear(4096, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),

                         nn.Linear(4096, 10))


if __name__ == '__main__':
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # (num_convs, outchannels)
    net = vgg(conv_arch)
    # 查看网络架构
    X = torch.randn(1, 1, 224, 224)
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)

    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]  # 简化
    net = vgg(small_conv_arch)
    lr, num_epochs, batch_size = 0.05, 10, 64
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    demo6_6.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
