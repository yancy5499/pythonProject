import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l.torch import Animator  # 画图类，详见d2l书本140页解释
import d2l.torch as d2l

'''
softmax从零实现
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


def softmax(X):
    """每个元素从x变为exp(x)，然后以每行为总量进行归一化(一行为一次预测，类型概率和为1)"""
    # 未考虑数值溢出
    X_exp = torch.exp(X)
    line_sum = X_exp.sum(1, keepdim=True)  # 结果为一个列向量
    return X_exp / line_sum


def net(X):
    """net(X)=XW+b"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    """交叉熵的信息量-log(p),在(0,1)内，越靠近0损失越大，越靠近1则损失越小"""
    # y为每列的正确类别1*n的列向量
    # 用range(len(y_hat)),y做索引，可以取出y_hat中对应位置的概率
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # 维数大于1且第二个维度大于1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 获取每行最大值的索引
        y_hat = y_hat.argmax(axis=1)
    # 将y_hat的type变成与y一致之后，再进行==逻辑运算
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上的模型精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 评估模式
    metric = Accumulator(2)  # 记录正确预测数和预测总数的类
    # metric=[0.0,0.0]
    with torch.no_grad():
        for X, y in data_iter:
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


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个epoch"""
    if isinstance(net, torch.nn.Module):
        net.train()  # 训练模式
    metric = Accumulator(3)
    for X, y in train_iter:
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


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))  # 画图
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    # 启动一个无梯度模式，在该模式中，无论输入是否requires_grad=True，都会按照False计算，以节省内存
    with torch.no_grad():
        for param in params:
            # 修正权重与偏置，学习率为lr
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def updater(batch_size):
    lr = 0.1
    return sgd([W, b], lr, batch_size)


if __name__ == '__main__':
    batch_size = 256
    # 加载数据集
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    num_inputs = 28 * 28
    num_outputs = 10
    # 权重与偏置
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    # 初始输入
    X = torch.arange(1, 7, dtype=torch.float32).reshape(2, -1)
    print(X, '\n',
          X.sum(0, keepdim=True), '\n',
          X.sum(1, keepdim=True))
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()

'''
1.在本节中，我们直接实现了基于数学定义softmax运算的softmax函数。这可能会导致什么问题？提示：尝试计算 \exp(50) 的大小。
如果网络参数初始化不恰当，或者输入有数值较大的噪音，基于数学定义的softmax运算可能造成溢出（结果超过long类型的范围）

2.本节中的函数 cross_entropy 是根据交叉熵损失函数的定义实现的。这个实现可能有什么问题？提示：考虑对数的值域。
y_hat中若某行最大的值也接近0的话，loss的值会超过long类型范围。

3.你可以想到什么解决方案来解决上述两个问题？
设定一个阈值，使得loss处于long类型能表达得范围内。

4.返回概率最大的标签总是一个好主意吗？例如，医疗诊断场景下你会这样做吗？
返回最大概率标签不总是个好主意。

5.假设我们希望使用softmax回归来基于某些特征预测下一个单词。词汇量大可能会带来哪些问题?
词汇量大意味着class的类别很多，这容易带来两个问题。一是造成较大的计算压力（矩阵计算时间复杂度O(n^2)。 二是所有的单词所得概率容易很接近0，单词间概率差别不大，很难判断应该输出哪个结果。
'''
