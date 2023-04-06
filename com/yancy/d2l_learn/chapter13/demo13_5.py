import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


def display_anchors(fmap_w, fmap_h, size):
    fig = plt.imshow(img)
    # 前两个维度不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=size, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(fig.axes,
                    anchors[0] * bbox_scale)
    plt.show()


img = plt.imread('../imgs/catdog.jpg')
h, w = img.shape[:2]  # img.shape:[h, w, channels]
print(img.shape, h, w)
display_anchors(fmap_w=4, fmap_h=4, size=[0.15])
display_anchors(fmap_w=2, fmap_h=2, size=[0.4])
display_anchors(fmap_w=1, fmap_h=1, size=[0.8])
