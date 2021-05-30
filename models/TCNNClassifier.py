"""
    Documentation: https://www.quora.com/What-are-temporal-convolutional-neural-networks
"""

import torch.nn as nn
from typing import Dict
from torch import Tensor
import torch

from helpers import Reshape


class TCNClassifier(nn.Module):
    def __init__(self, cfg: Dict):
        super(TCNClassifier, self).__init__()
        self.in_channels = int(cfg['tcn']['in_channels'])
        self.out_channels = int(cfg['tcn']['out_channels'])
        self.kernel_size = int(cfg['tcn']['kernel_size'])
        self.stride = int(cfg['tcn']['stride'])
        self.num_classes = int(cfg['system']['num_classes'])

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size,),
                stride=(self.stride,)
            ),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size,),
                stride=(self.stride,)
            ),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size,),
                stride=(self.stride,)
            ),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            Reshape(self.out_channels),
            nn.utils.weight_norm(nn.Linear(self.out_channels, self.num_classes)),
            nn.Softmax(dim=1)
        )

    def forward(self, data: Tensor) -> Tensor:
        return self.net(data)

    @property
    def device(self):
        return next(self.parameters()).device


def run_tcn_test():
    config = {
        'tcn': {
            'in_channels': 2,
            'out_channels': 8,
            'kernel_size': 2,
            'stride': 2
        },
        'system': {
            'num_classes': 10,
            'batch_size': 100
        }
    }

    c = TCNClassifier(cfg=config)
    print(c)
    data = torch.randn(size=(100, 2, 8))
    print('data', data.shape)
    out = c(data)
    print(out.shape)


if __name__ == '__main__':
    run_tcn_test()
