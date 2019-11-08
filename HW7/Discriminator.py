import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3,196,3,padding=1)
        self.ln1 = nn.LayerNorm([32,32])
        self.conv2 = nn.Conv2d(196,196,3,padding=1,stride=2)
        self.ln2 = nn.LayerNorm([16,16])
        self.conv3 = nn.Conv2d(196,196,3,padding=1)
        self.ln3 = nn.LayerNorm([16,16])
        self.conv4 = nn.Conv2d(196,196,3,padding=1,stride=2)
        self.ln4 = nn.LayerNorm([8,8])
        self.conv5 = nn.Conv2d(196,196,3,padding=1)
        self.ln5 = nn.LayerNorm([8,8])
        self.conv6 = nn.Conv2d(196,196,3,padding=1)
        self.ln6 = nn.LayerNorm([8,8])
        self.conv7 = nn.Conv2d(196,196,3,padding=1)
        self.ln7 = nn.LayerNorm([8,8])
        self.conv8 = nn.Conv2d(196,196,3,padding=0,stride=2)
        self.ln8 = nn.LayerNorm([4,4])
        self.pool = nn.MaxPool2d(4,stride=4)
        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    def forward(self,x):
        x = F.leaky_relu(self.ln1(self.conv1(x)))
        x = F.leaky_relu(self.ln2(self.conv2(x)))
        x = F.leaky_relu(self.ln3(self.conv3(x)))
        x = F.leaky_relu(self.ln4(self.conv4(x)))
        x = F.leaky_relu(self.ln5(self.conv5(x)))
        x = F.leaky_relu(self.ln6(self.conv6(x)))
        x = F.leaky_relu(self.ln7(self.conv7(x)))
        x = F.leaky_relu(self.ln8(self.conv8(x)))
        x = self.pool(x)
        x1 = self.fc1(x)
        x2 = self.fc10(x)
        return [x1,x2]
