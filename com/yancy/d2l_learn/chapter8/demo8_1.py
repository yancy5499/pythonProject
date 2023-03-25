import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
序列模型
'''


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    net.apply(init_weights)
    return net


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


def one_or_multi_pred(savefig=False):
    # 原数据
    plt.plot(time, x, label='data', linestyle='-')
    # 单步预测，如果知道前tau个时刻的信息，推导下一个信息
    onestep_preds = net(features)
    plt.plot(time[tau:],
             onestep_preds.detach().numpy(),
             label='onestep preds',
             linestyle=':')
    # 多步预测，如果不知道前tau个时刻的信息，根据已知的数据，逐步推导到前tau个信息，然后预测下一个信息
    multistep_preds = torch.zeros(T)  # 创建T长度的空张量
    multistep_preds[:n_train + tau] = x[:n_train + tau]  # 前n_trian+tau的数据已知
    for i in range(n_train + tau, T):
        # 从n_train+tau后开始，用预测值进行预测，多步推导
        multistep_preds[i] = net(multistep_preds[i - tau:i].reshape(1, -1))
    plt.plot(time[n_train + tau:],
             multistep_preds[n_train + tau:].detach().numpy(),
             label='multistep preds',
             linestyle='-.')
    plt.legend()
    if savefig:
        plt.savefig('./demo8-1-one_or_multi_pred.svg', dpi=1000, format='svg')
    plt.show()


def kstep_pred(savefig=False):
    # 给定tau个信息，预测后k个信息（前面的单步只预测一个信息，多部是不断单步的推导）
    max_steps = 64
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    for i in range(tau):
        # 前tau是已知值
        features[:, i] = x[i: i + T - tau - max_steps + 1]
    for i in range(tau, tau + max_steps):
        # tau后的是预测值，比如输入[[t1,t2,t3,t4],...]，返回了[t5,t6,...]，然后将返回值放到feature的右边连接起来
        # feature会变为[[t1,t2,t3,t4,t5'.t6',...],...]
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)
    steps = (1, 4, 16, 64)
    for i in steps:
        # 总共(T - max_steps)-(tau - 1)个样本，与i无关
        # 输出根据i变化，当i为1时，每个样本（四个信息）都预测下一个信息
        # 当i为4时，同样每个样本已知四个信息，根据这四个信息推导出下一个信息，再根据(3真1预测)的数据预测下一个信息，
        # 直到预测到第四个为止，以第四个预测点作为输出
        # in=[t1,t2,t3,t4]即样本(0:tau-1)，逐步推导出out=[t5',t6',t7',t8']即(tau:tau+i-1)
        # 最后输出最后一个out[-1]=features[:, tau + i - 1]
        plt.plot(time[tau + i - 1:T - max_steps + i],
                 features[:, tau + i - 1].detach().numpy(), label='{}-step preds'.format(i))
    plt.legend()
    if savefig:
        plt.savefig('./demo8-1-kstep_pred.svg', dpi=1000, format='svg')
    plt.show()


if __name__ == '__main__':
    T = 1000
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, size=(T,))  # 加入噪声
    plt.plot(time, x)
    plt.show()

    tau = 4
    features = torch.zeros((T - tau, tau))  # 数据一共有T-tau个样本，每个样本，带有自己以及前三个时间的数据，即4个
    # 不能做到T个样本，因为每个样本必须带有前三个状态的信息，只有从第四个开始才能满足样本要求，即T个数据只能做T-tau个样本
    for i in range(tau):
        # 将T个数据按列生成样本
        features[:, i] = x[i:T - tau + i]  # 样本如下:[[t1,t2,t3,t4],[t2,t3,t4,t5],...]
    labels = x[tau:].reshape((-1, 1))  # tau以后的数据都作为标签值，即正确答案
    batch_size, n_train = 16, 600  # 只使用数据的前600个训练模型，该模型可以预测T-tau个时间的结果
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                                batch_size, is_train=True)

    loss = nn.MSELoss(reduction='none')
    net = get_net()
    train(net, train_iter, loss, 5, 0.01)
    # 单步/多步预测
    # one_or_multi_pred()

    # k步预测
    kstep_pred()
