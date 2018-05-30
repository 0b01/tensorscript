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
