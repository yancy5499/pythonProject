import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter6.demo6_6 as demo6_6

'''
NiN网络
'''


# 放弃了全连接层，用NiN块取代，减少了参数量，但有时增加了训练时间
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )


if __name__ == '__main__':
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        # 输出标签数10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 四维输出转化为二维, size=(batch_size,10)
        nn.Flatten()
    )
    X = torch.rand(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.1, 10, 64
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    demo6_6.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
