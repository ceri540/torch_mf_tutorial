from statistics import variance
import torch
from  torch.autograd import Variable
import numpy as np

# variable
# torch 用tensor来计算
# 但是在神经网络中，实际上是将tensor放到变量中去进行神经网络计算
# 然后不断调整
# tensor是鸡蛋，variable是筐

tensor = torch.FloatTensor([[1, 2],[3, 4]])
variable = Variable(tensor, requires_grad = True)
# requires_grad是要不要将这个节点涉及到反向传播当中去
# print(tensor)
# print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
# print(t_out)
# print(v_out)

# 尝试查看这次计算中的误差的反向传递过程
# 用运算结果来反向传递，用输入来看梯度，这是基于他们是联系的
v_out.backward()
# 查看反向传递后的更新的梯度
# v_out = 1/4 * sum(var * var)
# d(v_out)/d(var) = 1/4 * 2 * variable = 2 * variable
# 这个公式就是梯度的计算公式

print(variable.grad)
print(variable.data)
print(variable.data.numpy())