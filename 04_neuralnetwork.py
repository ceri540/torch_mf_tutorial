from numpy import dtype
import torch
from torch.autograd import Variable

import torch.nn.functional as F
import matplotlib.pyplot as plt

# 生成标签数据和自变量数据，先定义维度
# 因为自变量包括x，y坐标，因此是一个二维变量,
# torch.normal是给定均值和标准差，然后生成一个矩阵
# 然后将自变量和标签拼接起来传递到篮子里
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
# 需要注意x，y的数据类型
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # float 32
y = torch.cat((y0, y1),).type(torch.LongTensor)     # integer 64

x, y = Variable(x), Variable(y)

# plt.scatter(
#     x.data.numpy()[:,0],
#     x.data.numpy()[:,1],
#     c=y.data.numpy(),
#     s=100,
#     lw=0)
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
net = Net(2,10,2)
# 分类的输出结果，如果是第零类
# 输出为[1,0]
# 否则为[1,0]
# n个自变量则类似
print(net)

# 可视化过程
# ion把这个过程变成一个实时打印的过程
plt.ion()
plt.show()
# 构建优化和误差
# optim中的lr就是学习的程度，lr低学的深，lr高学的快，一般取一个小于1的数即可
optimizer = torch.optim.SGD(net.parameters(),lr=0.002)
# 此处的误差函数应该选用使用分类的误差函数,多分类建议使用交叉熵
loss_func = torch.nn.CrossEntropyLoss()
# 因为需要注意的是，在分类问题中，输出结果的形式是概率值的形式出现
# 例如，二分类问题中，输出结果为[0.1, 0.9]即告诉我们分类是第一类的概率是0.9，第零类的概率是0.1
# 开始机器分类
for i in range(100):
    # 此时输出的还只是概率值
    out = net(x)

    loss = loss_func(out, y)

    # 每次循环都会将所有参数的梯度修改，因此在每次优化之前都要将梯度复位
    optimizer.zero_grad()
    # 然后进行反向传播，并且我们在前面variable中看到，反向传播之后都会伴随着梯度的变化
    loss.backward()
    # 然后以学习效率lr来优化这些梯度
    optimizer.step()

    if i%2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(
            x.data.numpy()[:,0],
            x.data.numpy()[:,1],
            c=pred_y,
            s=100,
            lw=0)
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Loss=%.2f'%accuracy, fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()