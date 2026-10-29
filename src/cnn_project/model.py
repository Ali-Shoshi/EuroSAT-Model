import numpy as np
import torch
import torch.nn as nn

class EuroSATModel(nn.Module):

    def __init__(self, channels = (16,32,64,128)):
        super(EuroSATModel, self).__init__()

        c1, c2, c3, c4 = channels

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=c3, out_channels=c4, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c4*4*4,10)
        )
    def forward(self,x):
        return self.model(x)

    def size(self):

        param_count = sum(p.numel() for p in self.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())

        size_mb = (param_bytes + buffer_bytes) / 1_000_000
        return param_count, size_mb

# Test the output shape of the model

# model=EuroSATModel()
# output= model(torch.randn(1,3,64,64))
# print(output.shape)
