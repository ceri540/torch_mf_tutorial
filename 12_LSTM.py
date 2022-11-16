## regression
## 算法流程 
## 1. 导入库
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


import numpy as np

# reproducible
torch.manual_seed(1)

## 2. 定义超参数
# Hyper params
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02
DOWNLOAD_MNIST = False


## 3. 准备数据
# prepare data
# data visualize -> 为了展示数据总量
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target(cos)')
# plt.plot(steps, x_np, 'b-', label='input(sin)')
# plt.legend(loc='best')
# plt.show()


## 4. 准备模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(                                  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=32,                                 # rnn hidden unit
            num_layers=1,                                   # number of rnn layer
            batch_first=True,                               # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):                          # 因为hidden state 是一直连续的，所以我们要不断传递这个参数
        # x shape (batch, time_step, input_size)
        # h_state shape (n_layers, batch, hidden_size)
        # r_out shape (n_layers, batch, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []                                           # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):              # 每一个时间点计算output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

    # rnn可以使用另外一种方式对每一个时间点求输出，
    # 上一种方法表现了torch动态构图的优势
    # def forward(self, x, h_state):
    #     r_out, h_state = self.rnn(x, h_state)
    #     r_out = r_out.view(-1, 32)
    #     outs = self.out(r_out)
    #     return outs.view(-1, 32, TIME_STEP), h_state

## 5. 引用模型
rnn = RNN()
print(rnn)


## 6. 定义optimization和loss function
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)       # optimize all rnn parameters
loss_func = nn.MSELoss()


## 7. 初始化h_state
h_state = None

## 8. 训练数据
# training and testing
for step in range(100):        
    start, end = step * np.pi, (step+1) * np.pi             # time step
    
    ## 8.1 输出本轮batch数据
    # sin 预测 cos -> 每次预测一部分来跑，相当于一个batch
    steps = np.linspace(start, end, 10, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    ## 8.2 运算数据并且计算pred和h_state
    prediction, h_state = rnn(x, h_state)
    # !!!
    h_state =  h_state.data                                 # 要h_state包装一下才可以放入下一个循环


    ## 8.3 反向传播和梯度下降最优化
    loss = loss_func(prediction, y)                         # cross entropy loss
    optimizer.zero_grad()                                   # clear gradients for this training step
    loss.backward()                                         # backpropagation, compute gradients
    optimizer.step()                                        # apply gradients


    ## 最优化
    # plot
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.5)

plt.ioff()
plt.show()