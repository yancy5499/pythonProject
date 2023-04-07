import os.path

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt
import com.yancy.d2l_learn.chapter13.demo13_6 as demo13_6
from com.yancy.d2l_learn.chapter3.demo3_7 import MyPlot


def cls_predictor(num_anchors, num_classes):
    # 类别预测层(每个锚框多一个背景类)
    return nn.LazyConv2d(num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_anchors):
    # 边界框预测层(每个框预测四个偏移量)
    return nn.LazyConv2d(num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)


def flatten_pred(pred):
    # 从1轴开始展平(将一个维度的所有元素连接在一起，展平至最后一维时元素已经是标量了就不能连接了)
    # 因此结果会保留0轴的元素数量，其余平铺至1轴
    # shape:(batch_size, h*w*channels)
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.concat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(out_channels):
    # 将输入的特征图高宽减半(类似VGG的block)
    blk = []
    for _ in range(2):
        # 不改变形状
        blk.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        blk.append(nn.LazyBatchNorm2d())
        blk.append(nn.ReLU())
    # 此处高宽减半
    blk.append(nn.MaxPool2d(2))  # 默认kernel_size=stride=2
    return nn.Sequential(*blk)


def base_net():
    # 基础网络块
    blk = []
    num_filters = [16, 32, 64]
    for i in num_filters:
        blk.append(down_sample_blk(i))
    return nn.Sequential(*blk)


def get_blk(i):
    if i == 0:
        return base_net()
    elif 1 <= i <= 3:
        return down_sample_blk(128)
    elif i == 4:
        return nn.AdaptiveAvgPool2d((1, 1))


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # self变量赋值语句的另一种写法, 后续的getattr是对应的取变量
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def test_net():
    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    # 计算损失的函数
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)
                   ).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,  # masks使负类锚框和填充锚框不参与损失的计算
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # 类别预测结果放在最后一维, 返回预测正确的数量
    return float(
        (cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum()
    )


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # 边界框的预测结果用平均绝对误差评价
    return float(
        (torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum()
    )


def train(net, train_iter, calc_loss, num_epochs, trainer, device):
    timer = d2l.Timer()
    fig = plt.figure()
    myplot = MyPlot(fig)
    net.to(device)
    print('start train_epoch...')
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels),  # 累积类别预测正确数
                       cls_labels.numel(),  # 累积类别标注数
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),  # 累积边界框偏移量的评分
                       bbox_labels.numel())  # 累积边界框标注数
            print(f'\repoch ing...[{epoch + 1}/{num_epochs}]', end='')
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        myplot.add_xy(epoch + 1, cls_err, bbox_mae)
    print(f'\nclass err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
          f'{str(device)}')
    myplot.show(labels=['class error', 'bbox mae'])


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


def display(img, output, threshold):
    plt.rcParams['figure.figsize'] = (5, 5)
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h),  # 边界框的左上角右下角坐标
                                        device=row.device)]
        d2l.show_bboxes(fig.axes, bbox,
                        '%.2f' % score, 'w')
    plt.show()


Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
print(concat_preds([Y1, Y2]).shape)
print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape)
print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)
print('=' * 10)
# ==========
# 0.2到1.05之间均匀划分成五分，每份长0.17，作为size的较小值
# 每个size的较大值由两个较小值的几何平均得出即sqrt(0.2*0.37)=0.272
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
# test_net()
# 读取数据集和初始化
batch_size = 32
train_iter, _ = demo13_6.load_data_bananas(batch_size)
device = d2l.try_gpu()
net = TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
# 定义损失函数
cls_loss = nn.CrossEntropyLoss(reduction='none')  # 分类损失
bbox_loss = nn.L1Loss(reduction='none')  # 偏移量损失（回归问题）
# 训练
num_epochs = 20
if os.path.exists('TinySSD.params'):
    print('查询到本地网络参数')
    # 读取参数文件
    params_dict = torch.load('TinySSD.params')
    # 根据网络架构重新建立网络，利用已知参数更新网络
    net.load_state_dict(state_dict=params_dict)
    net.to(device)
else:
    train(net, train_iter, calc_loss, num_epochs, trainer, device)
    torch.save(net.state_dict(), 'TinySSD.params')
'''
class err 3.31e-03, bbox mae 3.20e-03
7538.9 examples/sec on cuda:0
'''
# 获取样例图片
X = torchvision.io.read_image('../imgs/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
# 预测
output = predict(X)
# 展示
display(img, output.cpu(), threshold=0.9)
