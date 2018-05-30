import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Discriminator(nn.Module):
    '''Discriminator::forward([!1, <1>, <28>, <28>] -> [!4, <1>])'''
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(in_features=784, out_features=512)
        self.lin2 = nn.Linear(in_features=512, out_features=256)
        self.lin3 = nn.Linear(in_features=256, out_features=1)
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        x = F.leaky_relu(x)
        x = self.lin3(x)
        return F.sigmoid(x)


class Generator(nn.Module):
    '''Generator::forward([!1, <100>] -> [!1, <1>, <28>, <28>])'''
    def __init__(self):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(in_features=100, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.lin3 = nn.Linear(in_features=256, out_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.lin4 = nn.Linear(in_features=512, out_features=1024)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.lin5 = nn.Linear(in_features=1024, out_features=784)
    def forward(self, x):
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.lin3(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.lin4(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.lin5(x)
        x = F.tanh(x)
        return x.view(-1, 1, 28, 28)


