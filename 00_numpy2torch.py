import torch
import numpy as np

a_np = np.arange(6).reshape((2, 3))
a_torch = torch.from_numpy(a_np)

# print(a_np)
# print(a_torch)

a_torch2np = a_torch.numpy()

# print(a_torch2np)

# cal
data = [-1,-2,1,2]
data_tensor = torch.FloatTensor(data)   # float32
data_abs = torch.abs(data_tensor)
print(data_abs)
data_abs = np.abs(data)
print(data_abs)

data1 = [[1,2],[3,4]]
data1_tensor = torch.FloatTensor(data1)   # float32
data_np = np.array(data1)


print(
    '\nnumpy:', np.matmul(data_np,data_np),
    '\ntorch:', torch.mm(data1_tensor,data1_tensor)
)

# print(
#     '\nnumpy:', data_np.dot(data_np),
#     '\ntorch:', data1_tensor.dot(data1_tensor)
# )
# pytorch 更新之后吧dot操作只能够对1d数据进行处理，所以我们使用matmul
# 值得注意的是，在dot操作中，np中得到的是一个矩阵，但是tensor得到的实际上是一个值，因为他相当于aa^T