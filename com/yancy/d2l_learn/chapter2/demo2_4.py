import pylab
import matplotlib.pyplot as plt
import numpy as np

'''
函数与画图相关
'''


def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(5):
    print('h={:.5f}, numerical limit={:.5f}'.format(h, numerical_lim(f, 1, h)))
    h *= 0.1
# ===================
# 开始创建一个图像
plt.figure(figsize=(8, 5))
# 设置图像中的axe
ax = plt.gca()
# 隐藏右侧与上侧的脊柱
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# x轴的脊柱节点朝下
ax.xaxis.set_ticks_position('bottom')
# 底部脊柱设置位置
ax.spines['bottom'].set_position(('data', 0))
# y轴的脊柱节点朝左
ax.yaxis.set_ticks_position('left')
# 左侧脊柱设置位置
ax.spines['left'].set_position(('data', 0))
# 设置坐标轴
plt.xlim(-4.0, 4.0)
plt.xlabel('x', x=1)  # x轴名的位置位于100%处
plt.ylim(-2.0, 2.0)
plt.ylabel('f(x)', y=1)  # y轴名的位置位于100%处
# 曲线的变量空间
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# 设置图中曲线
plt.plot(X, np.cos(X),
         color='red',
         linewidth=2.0,
         linestyle='-',
         label='cos(x)')
plt.plot(X, np.sin(X),
         color='green',
         linewidth=2.0,
         linestyle='--',
         label='sin(x)')
# 创建图例，设置图例所在位置
plt.legend(loc='upper left')
plt.show()
# 一个图像结束
# ===================
plt.figure(figsize=(3.5, 2.5))
plt.xlim(-1, 4)
plt.xlabel('x')
plt.ylim(-5, 15)
plt.ylabel('f(x)')
X = np.linspace(0, 3, 256, endpoint=True)
plt.plot(X, f(X),
         color='red',
         linewidth=2.0,
         linestyle='-',
         label='f(x)')
plt.plot(X, 2 * X - 3,
         color='green',
         linewidth=2.0,
         linestyle='--',
         label='f\'(x))')
plt.legend(loc='upper left')
# 绘制网格
plt.grid()
plt.show()
# ===================
fig = plt.figure(figsize=(10, 10))
# 创建子图，划分为2行1列，其中axe1为左上角起第一个
axe1 = fig.add_subplot(2, 1, 1)
# 隐藏右侧与上侧的脊柱
axe1.spines['right'].set_color('none')
axe1.spines['top'].set_color('none')
# 底部脊柱设置位置
axe1.spines['bottom'].set_position(('data', 0))
# 左侧脊柱设置位置
axe1.spines['left'].set_position(('data', 0))
axe1.set_xlabel('x', x=1)
axe1.set_ylabel('f(x)', y=1)
X1 = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# 设置图中曲线
axe1.plot(X1, np.cos(X1),
          color='red',
          linewidth=2.0,
          linestyle='-',
          label='cos(x)')
axe1.plot(X1, np.sin(X1),
          color='green',
          linewidth=2.0,
          linestyle='--',
          label='sin(x)')
# 创建图例，设置图例所在位置
axe1.legend(loc='upper left')

axe2 = fig.add_subplot(2, 1, 2)
axe2.set_xlabel('x')
axe2.set_ylabel('f(x)')
X2 = np.linspace(0, 3, 256, endpoint=True)
axe2.plot(X2, f(X2),
          color='red',
          linewidth=2.0,
          linestyle='-',
          label='f(x)')
axe2.plot(X2, 2 * X2 - 3,
          color='green',
          linewidth=2.0,
          linestyle='--',
          label='f\'(x))')
axe2.legend(loc='upper left')
# 绘制网格
axe2.grid()
plt.show()
