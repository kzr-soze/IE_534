import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

no_of_hidden_units=196

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
        self.conv8 = nn.Conv2d(196,196,3,padding=1,stride=2)
        self.ln8 = nn.LayerNorm([4,4])
        self.pool = nn.MaxPool2d(4,stride=4)
        self.fc1 = nn.Linear(in_features=196,out_features=1)
        self.fc10 = nn.Linear(in_features=196,out_features=10)

    def forward(self,x, extract_features=0):
        x = F.leaky_relu(self.ln1(self.conv1(x)))
        if(extract_features==1):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, no_of_hidden_units)
            return h
        x = F.leaky_relu(self.ln2(self.conv2(x)))
        if(extract_features==2):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, no_of_hidden_units)
            return h
        x = F.leaky_relu(self.ln3(self.conv3(x)))
        if(extract_features==3):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, no_of_hidden_units)
            return h
        x = F.leaky_relu(self.ln4(self.conv4(x)))
        if(extract_features==4):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, no_of_hidden_units)
            return h
        x = F.leaky_relu(self.ln5(self.conv5(x)))
        if(extract_features==5):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, no_of_hidden_units)
            return h
        x = F.leaky_relu(self.ln6(self.conv6(x)))
        if(extract_features==6):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, no_of_hidden_units)
            return h
        x = F.leaky_relu(self.ln7(self.conv7(x)))
        if(extract_features==7):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, no_of_hidden_units)
            return h
        x = F.leaky_relu(self.ln8(self.conv8(x)))
        if(extract_features==8):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, no_of_hidden_units)
            return h
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = x.view(-1,196)
        # print(x.size())
        x1 = self.fc1(x)
        x2 = self.fc10(x)
        return [x1,x2]
