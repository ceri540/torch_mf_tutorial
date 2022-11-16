# mini batch running
import torch
import torch.utils.data as Data

# torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y) 
# loader = Data.DataLoader(
#     dataset=torch_dataset,
#     batch_size=Batch_size,
#     shuffle=True,
#     num_workers=2
# )
# 发生错误，错误为data_tensor是未预期的变量，解决方法如下:
# 新版的输入方法改为元祖，不需要定义直接对应输入，
# 要添加新的线程，要先定义main函数
# 另外值得注意的是，加了线程速度会变慢，因为GIL
# 也就是num_workers
def batch_main():
    Batch_size = 5

    x = torch.linspace(1, 10, 10)
    y = torch.linspace(10, 1, 10)

    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Batch_size,
        shuffle=True
    )
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print(
                'Epoch: ', epoch,
                '|Step: ', step,
                '|batch x: ', batch_x.numpy(),
                '|batch y: ', batch_y.numpy()
            )
if __name__=='__main__':
    batch_main()