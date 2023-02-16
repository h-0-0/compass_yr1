import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

class FC_FF_NN(nn.Module):
    #TODO: add ability to change number of layers and number of neurons per layer, and activation function
    def __init__(self):
        super(FC_FF_NN, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # Perform a forward pass of our model on some input
    def forward(self, x):
        x = self.flatten(x)
        out = self.net(x)
        return out