import torch

def save_module(x, y):
    net1 = torch.nn.Sequential(
        torch.nn.Linear(2, 10), 
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2)
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # save all the net
    torch.save(net1, 'net1.pkl')
    # save all the params
    torch.save(net1.state_dict(), 'net1_params.pkl')

def restore_net():
    net2 = torch.load("net1.pkl")
    return net2

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2)
    )
    net3.load_state_dict(torch.load('net1_params.pkl'))
    return net3