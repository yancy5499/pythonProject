import os
import pandas as pd
import torch

'''
以pandas包为例的数据集处理基础操作
'''
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
data_file = os.path.join('..', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# data.iloc(a,b)方法:序列a行与b列交叉的内容
# data.iloc(a:b,c:d)方法:序列a到b-1行与序列c到d-1列交叉的内容
# data.iloc([a,b],[c,d])方法:序列a和b行与序列c和d列交叉的内容，
inputs = inputs.fillna(inputs.mean())  # 用均值填补NaN
print(inputs)
# 将输入中的类别值（非数值），化成列数为类别数的01列
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
# 至此，data的全部元素均为数值类型
# pandas的data.values可以将表格内的内容转化为numpy数组(不包括序号列)
print(inputs.values)
X = torch.tensor(inputs.values)  # 输入矩阵
y = torch.tensor(outputs.values)  # 输出向量
print(X)
print(y)

# 课后习题
print('=' * 10)
data_test = pd.read_csv(data_file)
# print(data_test)
# print(data_test.isna())  # 对于dataFrame类型，判断其元素是否为nan，返回一个同格式的dataFrame真值表
nan_sum = data_test.isna().sum()  # pandas的sum函数默认计算每一列的和
# print(nan_sum)
# print(nan_sum.values)  # 转化为numpy数组
nan_max_index = nan_sum.values.argmax()
# print(nan_max_index)
# 最多nan的列名
col_name = data_test.columns.values[nan_max_index]
data_test_new = data_test.drop(columns=col_name)
# pandas的drop可以指定删除某行或某列，但行只能用index=索引，列只能用columns=列名
print(data_test_new)
