import torch
from torch import nn
import com.yancy.d2l_learn.chapter3.demo3_7 as demo3_7

'''
多层感知器简洁实现
'''


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(28 * 28, 2 ** 8),
                        nn.ReLU(),
                        nn.Linear(2 ** 8, 10))
    # net = nn.Sequential(nn.Flatten(),
    #                     nn.Linear(28 * 28, 2 ** 8),
    #                     nn.ReLU(),
    #                     nn.Linear(2 ** 8, 2 ** 7),
    #                     nn.ReLU(),
    #                     nn.Linear(2 ** 7, 10))
    net.apply(init_weights)
    # device = demo3_7.choose_device()
    # net.to(device)
    batch_size, lr, num_epochs = 256, 0.1, 20
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = demo3_7.load_data_fashion_mnist(batch_size)
    demo3_7.train(net, train_iter, test_iter, loss, num_epochs, trainer)  # device默认cpu
'''
一层隐藏层
over>>train_loss:0.3812
train_acc:0.8649
test_acc:0.8338
两层隐藏层,epoch=10
over>>train_loss:0.3968
train_acc:0.8558
test_acc:0.8340
两层隐藏层,epoch=20
over>>train_loss:0.3111
train_acc:0.8862
test_acc:0.8507
'''
