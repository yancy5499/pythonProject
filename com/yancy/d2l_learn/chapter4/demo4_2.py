import torch
from torch import nn
import com.yancy.d2l_learn.chapter3.demo3_7 as demo3_7

'''
多层感知器从零实现
'''


def relu(X):
    a = torch.zeros_like(X)
    # 按relu定义手动实现
    return torch.max(X, a)


def net(X, device=torch.device('cpu')):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # @代表矩阵乘法
    return H @ W2 + b2


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = demo3_7.load_data_fashion_mnist(batch_size)
    # 选择2的n次作为层的宽度
    num_inputs, num_outputs, num_hiddens = 28 * 28, 10, 2 ** 7
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True
    ) * 0.01)  # 乘以0.01是为了防止随机数过大
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True
    ) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]
    loss = nn.CrossEntropyLoss(reduction='none')
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    demo3_7.train(net, train_iter, test_iter, loss, num_epochs, updater)
'''
when num_hiddens=2**7, num_epoch=10:
over>>train_loss:0.3865
train_acc:0.8626
test_acc:0.8536

when num_hiddens=2**8, num_epoch=10:
over>>train_loss:0.3864
train_acc:0.8634
test_acc:0.8476

when num_hiddens=2**9, num_epoch=10:
over>>train_loss:0.3781
train_acc:0.8672
test_acc:0.8536

when num_hiddens=2**10, num_epoch=10:
over>>train_loss:0.3724
train_acc:0.8689
test_acc:0.8483

when num_hiddens=2**10, num_epoch=12:
over>>train_loss:0.3574
train_acc:0.8738
test_acc:0.8428
随着num_hiddens的增多，train逐渐过拟合，test不升反降
'''
