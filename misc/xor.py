import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Xor(nn.Module):
    '''Xor::forward([!1, <2>] -> [!1, <1>])'''
    def __init__(self):
        super(Xor, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=3)
        self.fc2 = nn.Linear(in_features=3, out_features=1)
    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        return self.fc2(x)

net = Xor()

inputs = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]))
targets = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0],
    [1],
    [1],
    [0]
]))

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

EPOCHS_TO_TRAIN = 20000
print("Training loop:")
for idx in range(0, EPOCHS_TO_TRAIN):
    for input, target in zip(inputs, targets):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # update
    if idx % 5000 == 0:
        print("loss: ", loss.data.numpy())
        print("Epoch: ", idx)


print("")
print("Final results:")
for input, target in zip(inputs, targets):
    output = net(input)
    print("Input:[{},{}] Target:[{}] Predicted:[{}] Error:[{}]".format(
        int(input.data.numpy()[0][0]),
        int(input.data.numpy()[0][1]),
        int(target.data.numpy()[0]),
        round(float(output.data.numpy()[0]), 4),
        round(float(abs(target.data.numpy()[0] - output.data.numpy()[0])), 4)
    ))
