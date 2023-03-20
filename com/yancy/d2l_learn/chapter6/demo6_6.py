import torch
import numpy as np
from torch import nn
from d2l import torch as d2l
from com.yancy.d2l_learn.chapter3.demo3_7 import MyPlot
import matplotlib.pyplot as plt

'''
卷积神经网络LeNet尝鲜
'''


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使⽤GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
    if not device:
        device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 初始化权重
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # =====画图相关=====
    fig = plt.figure()
    x_values = np.linspace(1, num_epochs + 1, num_epochs)
    my_plot = MyPlot(fig, x_values)
    # =====画图相关=====
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print('epoch{}>>train_loss:{:.4f}  train_acc:{:.4f}  test_acc:{:.4f}\n'
              .format(epoch + 1, train_l, train_acc, test_acc))
        # =====画图相关=====
        my_plot.add_y(train_l, train_acc, test_acc)
        # =====画图相关=====
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    # =====画图相关=====
    my_plot.show(labels=['train_loss', 'train_acc', 'test_acc'])
    # =====画图相关=====


if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Flatten(),  # 图像平铺，准备进行MLP

        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),

        nn.Linear(120, 84),
        nn.Sigmoid(),

        nn.Linear(84, 10)
    )
    X = torch.rand((1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    print('=' * 10)
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
