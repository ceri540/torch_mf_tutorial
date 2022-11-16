from turtle import forward
from numpy import dtype
import torch
from torch.autograd import Variable

import torch.nn.functional as F
import matplotlib.pyplot as plt

# unsqueeze可以将维度的数据转化，此处会将linspace生成的1d数据变成2d
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

# 创建一个神经网络，继承父类torch.nn.Module
# init和forward是最需要的
# init包含搭建层的信息
# forward是将init包含的信息搭建起来
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
net = Net(1,10,1)
print(net)

# 可视化过程
# ion把这个过程变成一个实时打印的过程
plt.ion()
plt.show()
# 构建优化和误差
# optim中的lr就是学习的程度，lr低学的深，lr高学的快，一般取一个小于1的数即可
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss_func = torch.nn.MSELoss()

# 开始机器回归
for i in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    # 每次循环都会将所有参数的梯度修改，因此在每次优化之前都要将梯度复位
    optimizer.zero_grad()
    # 然后进行反向传播，并且我们在前面variable中看到，反向传播之后都会伴随着梯度的变化
    loss.backward()
    # 然后以学习效率lr来优化这些梯度
    optimizer.step()

    if i%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        # plt.text(0.5, 0, 'Loss=%.4f'%loss.data[0], fontdict={'size':20,'color':'red'})
        # # invalid index of a 0-dim tensor. 
        # # Use `tensor.item()` in Python or 
        # # `tensor.item<T>()` in C++ to 
        # # convert a 0-dim tensor to a number
        plt.text(0.5, 0, 'Loss=%.4f'%loss.item(), fontdict={'size':20,'color':'red'})
        plt.pause(0.2)

plt.ioff()
plt.show()