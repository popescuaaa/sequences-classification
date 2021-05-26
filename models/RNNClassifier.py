import torch.nn as nn
from typing import Dict
from torch import Tensor


class RNNClassifier(nn.Module):
    def __init__(self, cfg: Dict):
        super(RNNClassifier, self).__init__()
        self.dim_input = int(cfg['rnn']['input'])
        self.dim_hidden = int(cfg['rnn']['hidden'])
        self.num_layers = int(cfg['rnn']['num_layers'])
        self.num_classes = int(cfg['system']['num_classes'])

        self.rnn = nn.RNN(input_size=self.dim_input,
                          hidden_size=self.dim_hidden,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.ll = nn.Linear(self.dim_hidden, self.num_classes)

    def forward(self, seq: Tensor) -> Tensor:
        r_out, (h_n, h_c) = self.rnn(seq)
        out = self.ll(r_out[:, -1, :])
        return out

    @property
    def device(self):
        return next(self.parameters()).device
