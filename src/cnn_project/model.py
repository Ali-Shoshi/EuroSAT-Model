import numpy as np
import torch
import torch.nn as nn



class EuroSATModel(nn.Module):

    def __init__(self):
        super(EuroSATModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*2*2,10)
        )
    def forward(self,x):
        return self.model(x)


# Test the output shape of the model

# model=EuroSATModel()
# output= model(torch.randn(1,3,64,64))
# print(output.shape)
