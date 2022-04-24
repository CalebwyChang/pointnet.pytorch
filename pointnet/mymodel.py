from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from dgl.geometry import farthest_point_sampler


class PointTransformerLayer(nn.Module):
    def __init__(self, dim):
        self.q = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        attention = self.softmax(Q@K.transpose(1,2)/np.sqrt(self.dim_k))@V

        return x


class PointTransformerBlock(nn.Module):
    """
    linear
    point transformer layer
    linear
    """
    def __init__(self, in_dim, out_dim, transfor_dim):
        self.fc1 = nn.Linear(in_dim, transformer_dim)
        self.transformer_layer = PointTransformerLayer() # ????
        self.fc2 = nn.Linear(transformer_dim, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # apply ReLU after fc for now
        out = self.relu(self.fc1(x))
        out = self.transformer_layer(out)
        out = self.relu(self.fc2(x))
        out = out + x
        return out


class TransitionDown(nn.Module):
    """
    input: (x, p1)
    farthest point sampling
    kNN
    mlp
    local max pooling
    output: (y, p2)
    """
    def __init__(self, in_dim, out_dim, k=16):
        self.k = k

    def forward(self, x):
        npoints = 
        point_idx = farthest_point_sampler(x, npoints)

        return None

class ResidualPoint(nn.Module):
    def __init__(self, dim):
        self.mlp = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.mlp(x)
        out = self.bn(out)
        out = self.relu(out)
        self.mlp(out)
        out = self.bn(out)
        out = x + out
        out = self.relu(out)
        return out

class OurPointNet(nn.Module):
    def __init__(self):
        self.first_mlp = nn.Sequential(
            nn.Linear(3, 3),
            nn.Linear(3, 32)
        )
        self.lin32 = nn.Linear(32, 32)
        self.transformer = PointTransformerBlock()
        self.down = TransitionDown()
        self.lin64 = nn.Linear(64, 64)

        self.res = ResidualPoint()
        self.max_pool = nn.MaxPool1d()

        self.final_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 40)
        )
    
    def forward(self, x):
        out =self.first_mlp(x)

        out = self.lin32(out)
        out = self.transformer(out)
        out = self.lin32(out)
        out = self.down(out)
        out = self.lin64(out)
        out = self.transformer(out)

        out = self.res(out)
        out = self.max_pool(out)
        out = self.res(out)
        out = self.max_pool(out)

        out = self.final_mlp(out)       
        return out



