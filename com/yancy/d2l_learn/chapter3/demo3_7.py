import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

'''
softmax简洁实现
'''


def load_data_fashion_mnist(batch_size, resize=None):
    """整合3.5步骤"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)  # 接受一个列表，其中的所有步骤整合在一起
    # 加载训练集
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=False
    )
    # 加载测试集
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=False
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # 维数大于1且第二个维度大于1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 获取每行最大值的索引
        y_hat = y_hat.argmax(axis=1)
    # 将y_hat的type变成与y一致之后，再进行==逻辑运算
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.sum())


def evaluate_accuracy(net, data_iter, device):
    """计算在指定数据集上的模型精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 评估模式
    metric = Accumulator(2)  # 记录正确预测数和预测总数的类
    # metric=[0.0,0.0]
    with torch.no_grad():
        for X, y in data_iter:
            if device != torch.device('cpu'):
                X, y = X.to('cuda:0'), y.to('cuda:0')
            metric.add(accuracy(net(X), y), y.numel())
            # metric = [0.0+acc, 0.0+ynum]
    return metric[0] / metric[1]


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch(net, train_iter, loss, updater, device):
    """训练模型一个epoch"""
    if isinstance(net, torch.nn.Module):
        net.train()  # 训练模式
    metric = Accumulator(3)
    for X, y in train_iter:
        if device != torch.device('cpu'):
            X, y = X.to('cuda:0'), y.to('cuda:0')
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 内置的优化器与损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 自制的优化器与损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失与训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater, device=torch.device('cpu')):
    # =====画图相关=====
    fig = plt.figure()
    x_values = np.linspace(1, num_epochs, num_epochs)
    my_plot = MyPlot(fig, x_values)
    # =====画图相关=====
    """训练模型"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater, device)
        test_acc = evaluate_accuracy(net, test_iter, device)
        train_loss, train_acc = train_metrics
        # =====画图相关=====
        my_plot.add_y(train_loss, train_acc, test_acc)
        # =====画图相关=====
        print('epoch{}>>train_loss:{:.4f}  train_acc:{:.4f}  test_acc:{:.4f}\n'.format(epoch + 1, train_loss, train_acc,
                                                                                       test_acc))
    train_loss, train_acc = train_metrics
    # =====画图开始=====
    my_plot.show(labels=['train_loss', 'train_acc', 'test_acc'])
    # =====画图结束=====
    assert train_loss < 0.5
    assert 1 >= train_acc > 0.7
    assert 1 >= test_acc > 0.7
    print('over>>train_loss:{:.4f}\ntrain_acc:{:.4f}\ntest_acc:{:.4f}'.format(train_loss, train_acc, test_acc))


class MyPlot:
    """画图类"""

    def __init__(self, figure, x_values):
        self.figure = figure
        self.x_values = x_values
        self.y_dic = {}

    def add_y(self, *y_values):
        for i in range(len(y_values)):
            if self.y_dic.get(i) is None:
                self.y_dic[i] = []
            self.y_dic[i].append(y_values[i])

    def show(self, labels=None,show_scatter=False):
        for i in range(len(self.y_dic)):
            if show_scatter:
                # 是否将结点画出来
                plt.scatter(self.x_values, self.y_dic[i])
            if type(labels) == list:
                plt.plot(self.x_values, self.y_dic[i], label=labels[i])
                plt.legend()
            else:
                plt.plot(self.x_values, self.y_dic[i])
        plt.grid()
        plt.show()


def choose_device(i=0, no_gpu=False):
    """如果存在gpu，返回gpu(i)，否则返回cpu()"""
    if not no_gpu:
        if torch.cuda.device_count() >= i + 1:
            return torch.device('cuda:{}'.format(i))
    return torch.device('cpu')


if __name__ == '__main__':
    batch_size = 256
    learning_rate = 0.1
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(),  # 该层用于将输入的图片摊平成一个向量，再到第二层作为输入
                        nn.Linear(28 * 28, 10))  # 输入28*28的向量，输出长度为10的向量
    device = choose_device(no_gpu=True)
    net.to(device=device)
    # d2l教程p144，重新审视softmax的实现，注意数据的上溢和下溢
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    num_epochs = 10

    train(net, train_iter, test_iter, loss, num_epochs, trainer, device)
'''
over>>train_loss:0.4477
train_acc:0.8474
test_acc:0.8262
'''
