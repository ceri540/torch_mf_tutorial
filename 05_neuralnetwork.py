from numpy import dtype
import torch
from torch.autograd import Variable

import torch.nn.functional as F
import matplotlib.pyplot as plt

# method1-create neural network
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):             
        super(Net, self).__init__()
        # 多少个输入，隐藏层节点数，多少个输出
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predic = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predic(x)
        return x

# 构建网络的类，并查看网络
net1 = Net(2,10,2)
print(net1)

# method2-create neural network
net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)
print(net2)

# output:
# Net(
#   (hidden): Linear(in_features=2, out_features=10, bias=True)
#   (predic): Linear(in_features=10, out_features=2, bias=True)
# )
# Sequential(
#   (0): Linear(in_features=2, out_features=10, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=10, out_features=2, bias=True)
# )
# 区别解读
# 首先层的名字不同，这是因为在类中定义的时候给定了名称，因此索引直接用类的定义
# 第二是激活函数有无显示的区别，
# 这是因为第一个的relu使用F的定义，这对于神经网络来说就等同于引用了python的一个func
# 因此没有特别表明，而第二个的定义使用torch.nn，此时会将他当做一层
# 以上的定义方式在实际使用中没有区别