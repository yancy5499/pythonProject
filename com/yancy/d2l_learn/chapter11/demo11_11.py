import math
import torch
from torch import nn
from torch.optim import lr_scheduler
from d2l import torch as d2l
import matplotlib.pyplot as plt
from com.yancy.d2l_learn.chapter3.demo3_7 import MyPlot

"""学习率衰减"""


class SquareRootScheduler:
    """简单的学习率调度器"""

    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)


class FactorScheduler:
    """单因子调度器"""

    def __init__(self, factor=1.0, stop_factor_lr=1e-7, base_lr=0.1):
        # 更新因子
        self.factor = factor
        # 最低学习率
        self.stop_factor_lr = stop_factor_lr
        # 当前学习率
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr


def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))
    return model


def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler=None):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 初始化权重
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    # =====画图相关=====
    fig = plt.figure()
    x_values = torch.linspace(1, num_epochs, num_epochs)  # torch和numpy的linspace均可
    my_plot = MyPlot(fig, x_values)
    # =====画图相关=====
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        print('epoch{}>>train_loss:{:.4f}  train_acc:{:.4f}  test_acc:{:.4f}\n'
              .format(epoch + 1, train_l, train_acc, test_acc))
        # =====画图相关=====
        my_plot.add_y(train_l, train_acc, test_acc)
        # =====画图相关=====
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()
            else:
                # 自定义的调度器特殊处理
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    # =====画图相关=====
    my_plot.show(labels=['train_loss', 'train_acc', 'test_acc'])
    # =====画图相关=====


def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr


class CosineScheduler:
    """余弦调度器"""

    def __init__(self, max_update, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        # 预热(防止初始学习率太小而太慢，初始学习率太大而发散)
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        # 预热，从warmup_begin_lr线性增加
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                   * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                    self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def show_scheduler():
    """查看调度器曲线"""
    num_epochs = 30

    # scheduler = SquareRootScheduler(lr=0.1)
    # 单因子调度器
    # scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
    # d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
    # plt.show()

    # 多因子调度器
    # net = net_fn()
    # trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    # scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
    # d2l.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler)
    #                                     for t in range(num_epochs)])

    # 余弦调度器
    scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01, warmup_steps=5)
    d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
    plt.show()


def test_scheduler():
    num_epochs = 30
    loss = nn.CrossEntropyLoss()
    device = d2l.try_gpu()
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    net = net_fn()
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    # 15和30为分段点，gamma为衰减系数
    scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
    train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler)


if __name__ == '__main__':
    show_scheduler()
    # 训练
    # test_scheduler()
'''
loss 0.183, train acc 0.931, test acc 0.881
38877.2 examples/sec on cuda:0
'''
