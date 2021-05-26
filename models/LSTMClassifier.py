from typing import Dict
import torch.nn as nn
from torch import Tensor


class LSTMClassifier(nn.Module):
    def __init__(self, cfg: Dict):
        super(LSTMClassifier, self).__init__()
        self.dim_input = int(cfg['lstm']['input'])
        self.dim_hidden = int(cfg['lstm']['hidden'])
        self.num_layers = int(cfg['lstm']['num_layers'])
        self.num_classes = int(cfg['system']['num_classes'])

        self.rnn = nn.LSTM(input_size=self.dim_input,
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