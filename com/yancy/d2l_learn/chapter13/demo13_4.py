import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

"""
锚框相关知识
"""

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())  # 预测的偏移量，此处设为0
cls_probs = torch.tensor([[0] * 4,  # 背景概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫概率

img = plt.imread('../imgs/catdog.jpg')
h, w = img.shape[:2]
bbox_scale = torch.tensor((w, h, w, h))
fig = plt.imshow(img)
d2l.show_bboxes(fig.axes, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
plt.show()

# 重新读取图片
fig = plt.imshow(img)
# outputs:(batch_size, num_anchors, 6)
# 6列分别为(类索引，置信度，左上角x，左上角y，右下角x，右下角y)，坐标为标准化值，后续还原需要乘上图片的bbox_scale
outputs = d2l.multibox_detection(cls_probs.unsqueeze(dim=0),
                                 offset_preds.unsqueeze(dim=0),
                                 anchors.unsqueeze(dim=0),
                                 nms_threshold=0.5)
for i in outputs[0].detach().numpy():
    if i[0] == -1:
        # 类索引-1表示背景或被nms抑制的框
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])  # 根据索引int(i[0])取类别，再加上置信度连成字符串
    # 展示所有框
    d2l.show_bboxes(fig.axes,
                    [torch.tensor(i[2:]) * bbox_scale],  # i[2:]即预测框的相对坐标(除以图片宽高)
                    label)
plt.show()
