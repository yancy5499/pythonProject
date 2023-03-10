import torch

'''
有关张量tensor的基本数据操作
'''
x = torch.arange(12)
print(x)  # type为tensor
print(x.shape)  # 访问张量x的形状，长宽
print(x.numel())  # tensor的元素总数
print(x.sum())  # 元素累加和
# print(x.item()) #当tensor的元素为一个数时，可以用item()方法转化为标量
"tensor的三个量分别为(深度，向量长度，向量个数)"

x = x.reshape(3, 4)  # 改变tensor的形状，并且元素基于x不变
print(x)
print(x.shape)
print(x.numel())
"当使用-1索引时可自动计算所需的维度如x.reshape(3,-1)或x.reshape(-1,4)"
print(x.reshape(2, 3, -1))

y = torch.zeros((2, 3, 4))  # 全零tensor
print(y)

z = torch.ones((2, 3, 4))  # 全一tensor
print(z)

print(torch.randn(3, 3))  # 生成元素随机的tensor
print(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  # 利用python数组作为data生产tensor

"简单运算符：按元素一一运算"
x = torch.arange(4).reshape(2, -1)
print(x + x)
print(x * x)
print(x ** x)
print(torch.exp(x))
"当按元素运算时，两边的张量形状不同时，会自我复制扩展，然后再运算"
x = torch.arange(2).reshape(2, -1)
y = torch.arange(2).reshape(1, -1)
print(x + y)

"张量连接"
x = torch.arange(4, dtype=torch.float32).reshape(2, -1)
y = torch.randn(2, 2)
print(torch.cat((x, y), dim=0))  # 按0轴连接，增加向量数量
print(torch.cat((x, y), dim=1))  # 按1轴连接，增加向量长度

"逻辑运算"
x = torch.arange(4, dtype=torch.float32).reshape(2, -1)
y = torch.tensor([[1, 0], [0, 1]])
print(x == y)
print(x < y)

"索引同python"
# tensor的索引可以写全tensor[a,b,c],其中abc分别为0，1，2轴，也可以只写部分，默认从0数
# 如tensor[a,b]为0轴索引是a，1轴索引是b
# a可以为切片a1:a2
# 同时还可为数组类，叫做广播机制,输入[a,b],[c,d]，相当于分别输入a,c和b,d
# 输入[a,b],c，相当于分别输入a,c和b,c
print('=' * 10)
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A[torch.tensor([0, 1])])  # 可以用tensor作为索引，同列表[0,1]
print(A[0, 2])  # (0,2)元素
print(A[[0, 2]])  # 相当于A[a,None],a=[0,2],输出0行和2行
print(A[[0, 2], [1, 2]])  # (0,2)元素和(1,2)元素
print(A[[0, 2], 1])
