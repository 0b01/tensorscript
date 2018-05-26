import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Mnist(nn.Module):
    '''Mnist::forward([!1, <1>, <28>, <28>] -> [!1, <10>])'''
    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = Dropout2d(p=0.5)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        nn.init.normal_(std=1)
        nn.init.normal_(std=1)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size)
        x = F.relu()
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, kernel_size)
        x = F.relu()
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu()
        x = self.example(x)
        return F.log_softmax(x, dim)
    def example(self, x):
        x = self.fc2(x)
        return F.relu()


