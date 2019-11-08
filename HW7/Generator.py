import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

class Generator(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(100,196*4*4)
        self.bn0 = nn.BatchNorm2d(196)
        self.conv1 = nn.ConvTranspose2d(196,196,4,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(196)
        self.conv2 = nn.Conv2d(196,196,3,padding=1)
        self.bn2 = nn.BatchNorm2d(196)
        self.conv3 = nn.Conv2d(196,196,3,padding=1)
        self.bn3 = nn.BatchNorm2d(196)
        self.conv4 = nn.Conv2d(196,196,3,padding=1)
        self.bn4 = nn.BatchNorm2d(196)
        self.conv5 = nn.ConvTranspose2d(196,196,4,padding=1,stride = 2)
        self.bn5 = nn.BatchNorm2d(196)
        self.conv6 = nn.Conv2d(196,196,3,padding=1)
        self.bn6 = nn.BatchNorm2d(196)
        self.conv7 = nn.ConvTranspose2d(196,196,4,padding=1,stride=2)
        self.bn7 = nn.BatchNorm2d(196)
        self.conv8 = nn.Conv2d(196,3,3,padding=1)

    def forward(self,x):
        x = F.relu(self.bn0(self.fc1(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.tanh(self.conv8(x))
        return x
