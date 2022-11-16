from calendar import EPOCH
from sklearn.metrics import accuracy_score
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1               # train n times
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False  # True when did not have it

def main_func():
    train_data = torchvision.datasets.MNIST(
        root='./mnist',                                 # download folder
        train=True,                                     # train data
        transform=torchvision.transforms.ToTensor(),    # transform to tensor
        download=DOWNLOAD_MNIST
    )

    # print(train_data.train_data.size())
    # # torch.size([60000, 28, 28])
    # print(train_data.train_labels.size())
    # # torch.size(60000)
    # plt.imshow(train_data.train_data[0].numpy(), cmap= 'gray')
    # plt.show()

    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_data = torchvision.datasets.MNIST(
        root = './mnist',
        train = False
    )

    test_x = Variable(
        torch.unsqueeze(test_data.test_data, dim=1),
        volatile = True).type(torch.FloatTensor)[:2000]/255.
    test_y = test_data.test_labels[:2000]

    class csy_cnn(nn.Module):
        def __init__(self):
            super(csy_cnn, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(                      # -> 1 28 28 
                    in_channels=1,              # rgb is 3    
                    out_channels=16,            # 卷积核的特征数，即其高度
                    kernel_size=5,              # 卷积核的长和宽都是5
                    stride=1,                   # 步长
                    padding=2                   # 扩展图片以扫描边缘，padding = (kernel_size-1)/2
                ),                              # -> 16 28 28
                nn.ReLU(),                      # -> 16 28 28
                nn.MaxPool2d(
                    kernel_size=2,
                ),                              # -> 16 14 14
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(                      # -> 16 14 14
                    in_channels=16,    
                    out_channels=32,            
                    kernel_size=5,              
                    stride=1,                   
                    padding=2                   
                ),                              # -> 32 14 14
                nn.ReLU(),                      # -> 32 14 14
                nn.MaxPool2d(
                    kernel_size=2
                ),                              # -> 32 7 7
            )

            self.out = nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0),-1)            # 展平数据 (batch, 32*7*7)
            output = self.out(x)
            return output


    cnn = csy_cnn()
    # print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x,y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)

            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / test_y.size(0)
                print('Epoch: ', epoch, 
                '| train loas: %4f' % loss.item(), 
                '| test accuracy:', accuracy)

        test_output = cnn(test_x[:10])
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        print(pred_y)
        print(test_y[:10].numpy())


if __name__ == '__main__':
    main_func()