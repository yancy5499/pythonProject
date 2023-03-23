import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter6.demo6_6 as demo6_6

'''
批量规范化(Batch Normalization)
'''


# 从零实现
def batch_norm(X, gamma, beta, moving_mean, moving_var, epsilon, momentum):
    if not torch.is_grad_enabled():
        # 预测模式下选择传入的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + epsilon)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 二维卷积层, 输入的X为(batch_size,channels,pixel_x,pixel_y)
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下使用当前的均值方差
        X_hat = (X - mean) / torch.sqrt(var + epsilon)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非学习参数
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, epsilon=1e-5, momentum=0.9
        )
        return Y


if __name__ == '__main__':
    pooling_Layer = nn.AvgPool2d(kernel_size=2, stride=2)
    activation_Layer = nn.Sigmoid()
    LeNet_useMyBatchNorm = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4),
        activation_Layer,
        pooling_Layer,

        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4),
        activation_Layer,
        pooling_Layer,

        nn.Flatten(),  # 图像平铺，准备进行MLP，检查平铺后元素是多少，从而修改下一层的输入数量

        nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2),
        activation_Layer,

        nn.Linear(120, 84), BatchNorm(84, num_dims=2),
        activation_Layer,

        nn.Linear(84, 10)
    )
    LeNet = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6),
        activation_Layer,
        pooling_Layer,

        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16),
        activation_Layer,
        pooling_Layer,

        nn.Flatten(),  # 图像平铺，准备进行MLP，检查平铺后元素是多少，从而修改下一层的输入数量

        nn.Linear(16 * 4 * 4, 120), nn.BatchNorm1d(120),
        activation_Layer,

        nn.Linear(120, 84), nn.BatchNorm1d(84),
        activation_Layer,

        nn.Linear(84, 10)
    )
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 1, 10
    demo6_6.train_ch6(LeNet_useMyBatchNorm, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    # demo6_6.train_ch6(LeNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
从零实现(python实现，性能不佳)
loss 0.269, train acc 0.901, test acc 0.938
24929.1 examples/sec on cuda:0
'''

'''
简洁实现(速度更快)
loss 0.290, train acc 0.892, test acc 0.938
36969.1 examples/sec on cuda:0
'''
