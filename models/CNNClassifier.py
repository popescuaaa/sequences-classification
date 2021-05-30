import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, cfg: Dict):
        super(CNNClassifier, self).__init__()
        self.in_channels = int(cfg['cnn']['in_channels'])
        self.out_channels = int(cfg['cnn']['out_channels'])
        self.kernel_size = int(cfg['cnn']['kernel_size'])
        self.num_classes = int(cfg['system']['num_classes'])
        self.batch_size = int(cfg['system']['batch_size'])

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size, ),
                stride=(2, )
            ),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size, ),
                stride=(2, )
            ),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(self.kernel_size,),
                stride=(2,)
            ),
            nn.ReLU(),
        )

        self.ll = nn.Linear(self.out_channels, self.num_classes)

    def forward(self, data: Tensor):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = data.view(-1, self.out_channels)
        out = self.ll(data)
        out = F.softmax(out, dim=1)
        return out


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
    print(out)


if __name__ == '__main__':
    run_cnn_test()
