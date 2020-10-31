import torch.nn as nn

class cpuModel(nn.Module):

    def __init__(self, module):
        super(cpuModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)

