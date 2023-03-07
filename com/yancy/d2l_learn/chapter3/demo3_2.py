import random

import matplotlib.pyplot as plt
import torch

'''
线性回归从零实现
'''


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+ε(噪声)"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 输入
    y = torch.matmul(X, w) + b  # 理想输出
    y = y + torch.normal(0, 0.01, y.shape)  # 加入噪音
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 输入的features，输出的labels
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[:3], '\nlabel:', labels[:3])
plt.figure()
# 取features的第二列绘出散点图观察与labels的关系
plt.scatter(features[:, 1], labels, 1)
plt.show()


# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 打乱样本
    # 遍历num_examples次，step=batch_size即每次取一份数据
    for i in range(0, num_examples, batch_size):
        # 第i次的数据batch,从i位到i+batch_size位，如果已经到了结尾，无法取到i+batch_size，则取末位
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        # 以生成器模式输出,将输出batch_indices作为0轴索引(行索引)对应的元素
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
# 复习,tensor的索引可以写全tensor[a,b,c],其中abc分别为0，1，2轴，也可以只写部分，默认从0数
# 如tensor[a,b]为0轴索引是a，1轴索引是b
# a可为切片a1：a2，同时还可为数组类，即同时索引0轴的多个值
# A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(A[torch.tensor([0, 1])])
# print(A[[0, 2]])

# 初始化模型参数
# w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
w = torch.zeros(size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    """平方误差（平方损失）"""
    # 均方除以2是为了求导后常数系数为1，本质不会改变
    # 预测值y_hat与真实值y
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    # 启动一个无梯度模式，在该模式中，无论输入是否requires_grad=True，都会按照False计算，以节省内存
    with torch.no_grad():
        for param in params:
            # 修正权重与偏置，学习率为lr
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 开始训练
# 初始化超参数
lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期为3
net = linreg  # 确定网络
loss = squared_loss  # 确定损失函数
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # 一次epoch迭代中一次batch数据的小批量损失
        # l的形状是(batch_size,1),化为标量计算梯度
        l.sum().backward()
        # 进行梯度下降
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        # 计算一次epoch的总损失(以均值表示)
        train_l = loss(net(features, w, b), labels)
        print('epoch:{},loss:{}'.format(epoch + 1, train_l.mean()))
print('w的估计误差:{}'.format(true_w - w.reshape(true_w.shape)))
print('b的估计误差:{}'.format(true_b - b))
