import time

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

'''
softmax回归(分类)
单层、全连接,权重为矩阵,一组特征对应的输出为多个类别的预测值
'''

# 读取数据集
trans = transforms.ToTensor()  # 转化器
# 下载训练集(若数据集已经存在目录当中了,则将download关掉)
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=False
)
# 下载测试集
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=False
)

print('train数据量:{},test数据量:{}'.format(len(mnist_train), len(mnist_test)))
print('train集中单个数据的size:{}'.format(mnist_train[0][0].shape))


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    # 返回一个列表，该列表的元素由text_labels[int(i)]生成，i取自labels中
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scales=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scales, num_rows * scales)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # 数组扁平化
    for i, (axe, img) in enumerate(zip(axes, imgs)):
        # zip将axes和imgs打包成一个元组，然后enumerate给元组添加索引，并生成一个迭代器
        # 输出一个变量形如(i,(axe,img)),在此for循环中直接用三个变量接收
        if torch.is_tensor(img):
            # 若img是图片张量，则用对应的方法将其输出成图片
            axe.imshow(img.numpy())
        else:
            # 否则为PIL图片
            axe.imshow(img)
        # # 隐藏axe的轴线
        axe.axes.get_xaxis().set_visible(False)
        axe.axes.get_yaxis().set_visible(False)
        if titles:  # 非空
            axe.set_title(titles[i])
    return fig


# 显示数据集中前几个图像及其标签
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28),  # 数据集中图片的像素为28*28，取18个
            2, 9,  # 输出时的行列数
            titles=get_fashion_mnist_labels(y)).show()

# 读取小批量数据集
batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=0)  # 线程数为0，即不开多线程
start = time.time()
for X, y in train_iter:
    continue
end = time.time()
print('读取数据集花费了:{}s'.format(end - start))

test_iter = data.DataLoader(mnist_test, batch_size, shuffle=True,
                            num_workers=0)
for X, y in test_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    # 只读一步
    break
