import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
实战Kaggle比赛：预测房价
'''


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    # 神经网络误差指标
    # 预处理，将小于1的值设置为1，简化后续log运算
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 选择优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        # 将该epoch的网络误差存入列表
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_train], 0)
            y_train = torch.cat([y_train, y_train], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print('折{},训练log rmse {:.5f},验证log rmse {:.5f}'
              .format(i + 1, train_ls[-1], valid_ls[-1]))
        if i == 0:
            plt.figure()
            plt.xlabel('epoch')
            plt.ylabel('rmse')
            plt.yscale('log')
            x_values = list(range(1, num_epochs + 1))
            plt.plot(x_values, torch.tensor(train_ls), label='train_ls')
            plt.plot(x_values, torch.tensor(valid_ls), label='valid_ls', linestyle='--')
            plt.legend()
            plt.grid()
            plt.show()
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('log rmse')
    plt.yscale('log')
    plt.plot(np.arange(1, num_epochs + 1), torch.tensor(train_ls))
    plt.grid()
    plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将⽹络应⽤于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    DATA_HUB = dict()
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

    DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv',
                                      '585e9cc93e70b39160e7921475f9bcd7d31219ce')

    DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv',
                                     'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
    fname_train = d2l.download('kaggle_house_train')
    fname_test = d2l.download('kaggle_house_test')
    # pandas见demo2_2
    train_data = pd.read_csv(fname_train)
    test_data = pd.read_csv(fname_test)
    print('数据集大小:train={},test={}'.format(train_data.shape, test_data.shape))
    # 作为模型输入时需要剔除无信息的id列,且train和test连成一个表
    all_features = pd.concat((train_data.iloc[:, 1:-1],
                              test_data.iloc[:, 1:-1]))
    # 将所有数值标准化后，空值替换成对应特征的平均值
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 获取数值元素列的索引
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())  # 标准化,此后均值为0,方差为1
    )
    all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 均值0替换空值
    # object类别值转化为01数值
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print('调整后all_features\' shape:', all_features.shape)
    n_train = train_data.shape[0]
    # 通过pd.values转化为tensor,通过n_train分割表
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    # 价格作为标签,化为一维向量
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    # 训练
    loss = nn.MSELoss()
    # 输入特征的数量,每一行作为一个样本，列数为特征数量
    in_features = train_features.shape[1]
    k, num_epochs, lr, weight_decay, batch_size = 5, 200, 5, 0.09, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print('{}-折,平均训练log rmse {:.5f},平均验证log rmse {:.5f}'
          .format(k, train_l, valid_l))
    # train_and_pred(train_features, test_features, train_labels, test_data,
    #                num_epochs, lr, weight_decay, batch_size)
