import torch
from torch import nn
import com.yancy.d2l_learn.chapter3.demo3_7 as demo3_7
from d2l import torch as d2l

'''
dropout暂退法
'''


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # 生成[0,1]均匀分布,与dropout做逻辑大于运算，大于的为true=1，小于为false=0
    mask = torch.rand(X.shape) > dropout
    return mask * X / (1.0 - dropout)


class Net(nn.Module):  # 继承nn.Module
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.training = is_training
        self.num_inputs = num_inputs
        # 线性层
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print(X)
    print(dropout_layer(X, 0.0))
    print(dropout_layer(X, 0.5))
    print(dropout_layer(X, 1))
    print('=' * 10)
    # 模型中使用
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5
    num_epochs, lr, batch_size = 20, 0.1, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = demo3_7.load_data_fashion_mnist(batch_size)
    # 从零实现
    # net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    # 简洁实现
    net = nn.Sequential(
        # 平铺
        nn.Flatten(),
        # 进入第一个隐藏层
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(dropout1),
        # 进入第二个隐藏层
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(dropout2),
        # 输出层
        nn.Linear(num_hiddens2, num_outputs)
    )
    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), weight_decay=0.001, lr=lr)
    demo3_7.train(net, train_iter, test_iter, loss, num_epochs, trainer)
    # ==========
    # lr=0.5,decay=0,dropout
    # over >> train_loss: 0.2799
    # train_acc: 0.8961
    # test_acc: 0.8768

    # lr=0.1,decay=0.001,dropout
    # over >> train_loss: 0.3604
    # train_acc: 0.8717
    # test_acc: 0.8633
    # ==========
    '''
    net_without_dropout = nn.Sequential(
        # 平铺
        nn.Flatten(),
        # 进入第一个隐藏层
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        # 进入第二个隐藏层
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        # 输出层
        nn.Linear(num_hiddens2, num_outputs)
    )
    net_without_dropout.apply(init_weights)
    trainer = torch.optim.SGD(net_without_dropout.parameters(), lr=lr)
    demo3_7.train(net_without_dropout, train_iter, test_iter, loss, num_epochs, trainer)
    '''
    # ==========
    # lr=0.5,decay=0,without_dropout
    # over >> train_loss: 0.2505
    # train_acc: 0.9054
    # test_acc: 0.8661
    # ==========
    # 暂退法是引入一定的噪声，增加模型对输入数据的扰动鲁棒，从而增强泛化；
    # 权重衰减在于约束模型参数防止过拟合
    # 可以共同作用,但需要设置好dropout,learning_rate和decay值
    # 若欠拟合，需要减小正则项系数decay,减小learning_rate，后者要大两三个数量级
    # 若过拟合需要增加decay，增大learning_rate
