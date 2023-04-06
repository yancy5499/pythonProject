import torch
import torch.utils.data as data
import torchvision
from d2l import torch as d2l
import matplotlib.pyplot as plt
import os
import pandas as pd

"""
目标检测数据集
"""


def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    if os.path.exists('../data/banana-detection'):
        data_dir = os.path.join('../data/banana-detection')
    else:
        data_dir = d2l.download_extract('banana-detection')
        os.remove(os.path.join('../data', 'banana-detection.zip'))
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir,
                         'bananas_train' if is_train else 'bananas_val',
                         'images', f'{img_name}')))
        # 这⾥的target包含（类别，左上⻆x，左上⻆y，右下⻆x，右下⻆y），
        # 所有的图像都包含相同的香蕉类(index=0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = data.DataLoader(BananasDataset(is_train=True),
                                 batch_size, shuffle=True)
    val_iter = data.DataLoader(BananasDataset(is_train=False),
                               batch_size)
    return train_iter, val_iter


class BananasDataset(data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""

    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train
                                                   else f' validation examples'))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)


batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)
# 展示部分图片
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
plt.show()
