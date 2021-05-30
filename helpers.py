import torch.nn as nn
from torch import Tensor


class Logger(nn.Module):
    def __init__(self):
        super(Logger, self).__init__()

    def forward(self, data: Tensor) -> Tensor:
        print('Logger', data.shape)
        return data


class Reshape(nn.Module):
    def __init__(self, size: int):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, data: Tensor) -> Tensor:
        return data.view(-1, self.size)
