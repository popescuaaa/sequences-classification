import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor
from helpers import Reshape


class CNNClassifier(nn.Module):
    def __init__(self, cfg: Dict):
        super(CNNClassifier, self).__init__()
        self.in_channels = int(cfg['cnn']['in_channels'])
        self.out_channels = int(cfg['cnn']['out_channels'])
        self.kernel_size = int(cfg['cnn']['kernel_size'])
        self.num_classes = int(cfg['system']['num_classes'])
        self.batch_size = int(cfg['system']['batch_size'])

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size,),
                stride=(2,)
            ),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size,),
                stride=(2,)
            ),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size,),
                stride=(2,)
            ),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            Reshape(self.out_channels),
            nn.Linear(self.out_channels, self.out_channels * 3),
            nn.ReLU(),
            nn.Linear(self.out_channels * 3, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, data: Tensor):
        out = self.net(data)
        return out

    @property
    def device(self):
        return next(self.parameters()).device


def run_cnn_test():
    config = {
        'cnn': {
            'in_channels': 2,
            'out_channels': 8,
            'kernel_size': 2,
        },
        'system': {
            'num_classes': 10,
            'batch_size': 100
        }
    }

    c = CNNClassifier(cfg=config)
    print(c)
    data = torch.randn(size=(100, 2, 8))
    print(data.shape)
    out = c(data)
    print(out.shape)


if __name__ == '__main__':
    run_cnn_test()
