# optimizer 加速
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

def opt_select():
    torch.manual_seed(1)

    # 确定批处理基本参数
    Lr = 0.01
    Batch_size = 32
    Epoch = 12

    # 伪数据
    x = torch.unsqueeze(torch.linspace(-1, 1, 10000),dim=1)
    y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

    # plt.scatter(x.numpy(),y.numpy())
    # plt.show()

    # 定义模型
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(1,20)
            self.predic = torch.nn.Linear(20,1)

        def forward(self,x):
            x = F.relu(self.hidden(x))
            x = self.predic(x)
            return x

    net_SGD = Net()
    net_Momentum = Net()
    net_RMSProp = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum,net_RMSProp, net_Adam]

    # 定义最优化函数
    optimizer_SGD = torch.optim.SGD(net_SGD.parameters(),lr=Lr)
    optimizer_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=Lr, momentum = 0.8)
    optimizer_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(),lr=Lr, alpha=0.9)
    optimizer_Adam = torch.optim.Adam(net_Adam.parameters(), lr=Lr, betas=(0.9,0.99))
    optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_RMSProp, optimizer_Adam]

    # 定义误差
    loss_func = torch.nn.MSELoss()
    loss_his = [[], [], [], []]
    # 定义dataloader
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size=Batch_size,
        shuffle=True
    )
    # 训练
    for epoch in range(Epoch):
        print('Epoch: ', epoch)
        for step, (batch_x, batch_y) in enumerate(loader):
            # 遍历每个优化器，优化对应的神经网络
            for net, opt, l_his in zip(nets, optimizers, loss_his):
                out = net(batch_x)
                loss = loss_func(out, batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())
    
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(loss_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

if __name__ == '__main__':
    opt_select()