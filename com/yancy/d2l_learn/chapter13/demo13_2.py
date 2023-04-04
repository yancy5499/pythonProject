import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch.utils.data as data
from com.yancy.d2l_learn.chapter3.demo3_7 import MyPlot

"""
微调
"""


def show_dataset_example():
    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
    plt.show()


def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train(net, train_iter, test_iter, loss, trainer, num_epochs,
          devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    # =====画图=====
    fig = plt.figure()
    myplot = MyPlot(fig)
    # =====画图=====
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            print(f'\repoch ing...[{epoch + 1}/{num_epochs}], batch[{i + 1}/{num_batches}]', end='')
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        # =====画图=====
        myplot.add_xy(epoch + 1,
                      metric[0] / metric[2], metric[1] / metric[3], test_acc)
        # =====画图=====
    print(f'\nloss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    myplot.show(labels=['train loss', 'train acc', 'test acc'])


def train_fine_tuning(net, learning_rate, batch_size=64, num_epochs=5,
                      param_group=True):
    train_iter = data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    train(net, train_iter, test_iter, loss, trainer, num_epochs,
          devices)


if os.path.exists('../data/hotdog'):
    data_dir = os.path.join('../data/hotdog')
else:
    data_dir = d2l.download_extract('hotdog')
    os.remove(os.path.join('../data/hotdog.zip'))
# 读取数据集
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
show_dataset_example()
# 使用RGB通道的均值和方差标准化每个通道
normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 整合所有预处理步骤
train_augs = torchvision.transforms.Compose(
    [torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪后，缩放成224*224
     torchvision.transforms.RandomHorizontalFlip(),  # 随机水平旋转
     torchvision.transforms.ToTensor(),  # 生成张量
     normalize]  # 标准化
)
test_augs = torchvision.transforms.Compose(
    [torchvision.transforms.Resize([256, 256]),  # 缩放
     torchvision.transforms.CenterCrop(224),  # 中心裁剪出224*224
     torchvision.transforms.ToTensor(),
     normalize]
)
# 下载预训练模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 输出层fc
# print(pretrained_net.fc)
# 进行微调
pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, 2)
finetune_net = pretrained_net
# print(finetune_net.fc)
# 初始化输出层的权重
nn.init.xavier_uniform_(finetune_net.fc.weight)
# 训练(微调使用较小学习率)
train_fine_tuning(finetune_net, learning_rate=5e-5)
'''
loss 0.200, train acc 0.918, test acc 0.921
126.0 examples/sec on [device(type='cuda', index=0)]
'''
