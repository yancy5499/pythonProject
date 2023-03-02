import torch

'''
矩阵与向量运算等线代基础操作
'''
A = torch.arange(9).reshape(3, 3)
print(A)
print(A.T)  # 矩阵转置

# sum函数可用于降维，即指定某一轴求和,0行1列
print(A.sum(axis=1))  # 求和后统一会变成一个向量，即一行
print(A.sum(axis=1, keepdims=True))  # 可保持轴数，这样结果向量会变成列，便于后续处理
print(A / A.sum(axis=1, keepdims=True))
print('=' * 10)
print(A.sum(axis=0))  # 求和后统一会变成一个向量，即一行
print(A.sum(axis=0, keepdims=True))  # 可保持轴数，便于后续处理，如果是行求和，处理前后无区别
print(A / A.sum(axis=0, keepdims=True))
# mean求均值，也可指定轴
# cumsum函数类似sum,结果的形状不会变化，存放方法为[a,b,c].cumsum(axis=1)>>>[a,a+b,a+b+c]
# axis=1即以1轴为结果轴，求和
print('=' * 10)
print(A.cumsum(axis=1))
# 点积，用于向量，对应位置的元素相乘再累加
print('=' * 10)
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 2, 1])
print(x)
print(y)
print(torch.dot(x, y))  # 也可用torch.sum(x * y)实现
# 矩阵and向量:mv
# 矩阵and矩阵:mm
print('=' * 10)
x = torch.arange(4, dtype=torch.float32).reshape(2, -1)
print(x.dtype)
y = torch.tensor([1.0, 2.0])  # 在矩阵与向量运算时，向量视为竖的
z = torch.eye(2)
print(x)
print(y)
print(z)
print(torch.mv(x, y))
print(torch.mm(x, z))
# 范数，默认2范数
A = torch.arange(9, dtype=torch.float32).reshape(3, 3)  # 注意dtype，默认int无法计算
print(torch.norm(A))
# print(A.norm())  # 同上
print(A.norm(1))  # 全部元素绝对值之和
print(A.norm(2))  # 全部元素平方和的开方
print(A.norm(float('inf')))  # 全部元素绝对值的最大值
# A.norm(p) # 全部元素p次方和的1/p次方

# 练习
print('=' * 10)
print(len(torch.tensor([2, 3, 4])))
A = torch.arange(9).reshape(3, 3)
print(len(A))  # 结果为张量中向量的长度，即0轴长度
print('=' * 10)
A = A.reshape(1, 3, 3)
B = torch.cat((A, torch.ones(1, 3, 3)), dim=0)
# 这时候的0轴是深度，与定义tensor时相关，tensor二维时[0,1]分别为行列
# tensor为三维时[0,1,2]为深、行、列
print(A)
print(B)
print(B.norm())
