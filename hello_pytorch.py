from __future__ import print_function
import torch
# 创建一个没有初始化的5×3的矩阵
# x = torch.empty(5, 3)

# 创建一个随机初始化矩阵
# x = torch.rand(5, 3)

# 创建一个填满0而且数据类型为long的矩阵
# x = torch.zeros(5, 3, dtype=torch.long)

# 直接从数据构造张量
x = torch.tensor([5.5, 3])

# 根据已有的tensor建立新的tensor,创建一个填满1，大小为5×3, 数据类型为double的张量
x = x.new_ones(5, 3, dtype=torch.double)
# 根据已有的tensor建立新的tensor,除非用户提供新的值，否则将重用输入张量的属性，比如size
x = torch.randn_like(x, dtype=torch.float)  # 虽然没有指定size, 但新建的张量的size跟以前的那个是一致的
# randn_like和randn一样，都是返回均值为0,方差为1的正太分布中的随机数
# print(x)

# 获取张量形状
# print(x.size()) # 返回类型：tuple

'''
加法运算
'''
y = torch.rand(5, 3)
# 加法的第一种形式：直接加
# print(x+y)

# 加法的第二种形式：调用torch.add(a,b)函数
# print(torch.add(x, y))
# 特别的，还可以指定把相加的结果存储到某一个张量中
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)

# 加法的第三种形式：原地操作/in-place操作
# y.add_(x)   # y += x
# # 任何一个in-place改变张量的操作后面都固定一个_。例如x.copy_(y)、x.t_()将更改x
# print(y)

'''
索引
'''
# print(x)
# print(x[-1])    #  输出最后一行
# print(x[:, -1]) # 输出最后一列

'''
改变形状：torch.view
'''
x = torch.randn(4, 4)
y = x.view(16)
# print(x)
# print(y)
# print(y.size())

# z = x.view(-1, 8)   # -1表示不指定是几行，但只指定令8列。也就是说行数由程序自动算得
# print(x)
# print(z)
# print(z.size())

'''
对于只含一个元素的张量，可以用 .item()来得到对应的python数值
'''
# x = torch.randn(1)
# print(x)
# print(x.item())

'''
张量torch和numpy数组的互换
特别的， 二者是共享内存的。因此，当一个改变时，另外一个也会改变
'''
# # 把张量转为numpy数组
# a = torch.ones(5)
# print(a)
#
# b = a.numpy()
# print(b)
# # 若改变张量，那么数组也会跟着变化
# a.add_(1)
# print(a)
# print(b)

#  把numpy数组转为张量: tensorX = torch.from_numpy(numpyX)
import numpy as np
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
np.add(a, 1, out=a)
print(a)
print(b)

