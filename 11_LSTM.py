## classifying
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# hyper params
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28                                      # rnn time step equal wit image height
INPUT_SIZE = 28                                     # rnn input size equal with image width
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root='./mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# print(train_data.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()


train_loader = data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = dsets.MNIST(
        root = './mnist',
        train = False,
        transform=transforms.ToTensor()
    )

test_x = test_data.data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets.numpy()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = 64,
            num_layers = 1,                         # lstm 的隐藏层 越大能力越强时间越长
            batch_first=True                        # (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)
    
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)       # x的形状 (batch, time_step, input_size)
                                                    # h_n 分线层的隐藏层 h_c 主线层的隐藏层
                                                    # 初始节点无隐藏层输入，故为None
        out = self.out(r_out[:, -1, :])             # (batch, time_step, input_size) 取最后一个时间的
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1,28,28)                  # define x
        b_y = y                                 # define y

        output = rnn(b_x)                       # output y'
        loss = loss_func(output, b_y)           # caculate loss

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
                test_output = rnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y).astype(int).sum())
                print('Epoch: ', epoch, 
                '| train loas: %.4f' % loss.item(), 
                '| test accuracy: %.2f', accuracy)

test_output = rnn(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print(pred_y)
print(test_y[:10])