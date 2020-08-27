'''
PyTorch中，所有神经网络的核心是 autograd 包。 它可以为基于tensor的的所有操作提供自动微分的功能
autograd包中，最重要的两个类分别是torch.tensor(这个包的核心类)和function
torch.tensor类有一个重要的属性： .requires_grad, 可以设置是否追踪该张量的操作。.backward()方法可以计算所有的梯度
tensor和function互相连接形成令一个无圈图 acyclic graph, 它编码了完整的计算历史
'''

import torch
# 创建一个张量x, 并设置requires_grad=True来对于张量X的所有操作
x = torch.ones(2, 2, requires_grad=True)
# print(x)
#
y = x + 2
# print(y)    # 这里我们看到y具有grad_fn属性
'''关于grad_fn属性
除了用户手动创建的变量外，所有变量都有grad_fn属性，这个属性引用了一个创建variable的操作，如加减乘除
'''
# print(y.grad_fn)

z = y * y * 3
# print(z)    # ???grad_fn = MulBackward0是什么意思

out = z.mean()  # Returns the mean value of all elements in the input tensor
# print(out)  # ???grad_fn = MeanBackward是什么意思

'''
如果不指定某个张量的require_grad属性，则默认是False
'''
a = torch.randn(2, 2)
a = ((a*3) / (a-1))
# print(a)
# print(a.requires_grad)  # 由于之前没有指定a，所以这里输出的是false

a.requires_grad_(True)
# print(a.requires_grad)  # 由于之前我们指定令，所以这里输出的是true

b = (a*a).sum() # sum() eturns the sum of all elements in the input tensor.
# print(b)
# print(b.grad_fn)

'''梯度
'''
# print(out)  # out: tensor(27., grad_fn=<MeanBackward0>)
# # out.backward()
# print(out.backward())   # None
# '''backward() computes the sum of gradients of given tensors
# '''
# print(x)
# print(x.grad)   # d(out)/dx，我们通过以上代码计算出了out对x求导的值为4.5

# z.backward()  # 奇怪的是，z调用backward()时会报错，RuntimeError: grad can be implicitly created only for scalar标量 outputs

'''
雅可比向量积的链式特性 使得将外部梯度输入到具有非标量输出的模型中，变得非常方便
torch.autograd是计算雅可比向量积的一个引擎
'''
