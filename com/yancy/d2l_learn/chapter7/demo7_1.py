import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter6.demo6_6 as demo6_6

'''
AlexNet精简版
'''

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Flatten(),
        # 全连接层输出非常大，使用dropout防止过拟合
        nn.Linear(6400, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 10)
    )
    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    batch_size = 128
    # 增大图像分辨率，以测试AlexNet
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10
    demo6_6.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.325, train acc 0.881, test acc 0.938
446.5 examples/sec on cuda:0
'''
