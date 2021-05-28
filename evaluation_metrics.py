from __future__ import print_function

from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn as nn
from data import Pendigits
import torch
from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def cm(model: nn.Module, cfg: Dict, save_figure: bool = False, fig_name: str = None, show: bool = False):
    ds = Pendigits.Pendigits(file_path='./data/pendigits.tes')
    device = torch.device(cfg['system']['device'])

    data, labels = ds.get_all()
    data = torch.from_numpy(data)
    bs, seq_len, features = data.shape
    data = data.float()
    data = data.to(device)
    data = data.view(bs, features, seq_len)

    out = model(data)
    _, pred = torch.max(out, 1)

    pred = pred.detach().cpu().numpy()

    conf_matrix = confusion_matrix(labels, pred)

    print('Accuracy score: {:.6f}'.format(accuracy_score(labels, pred)))

    plt.figure(figsize=(10, 10), dpi=250)
    sns.heatmap(conf_matrix, annot=True)
    if show:
        plt.show()
    if save_figure:
        plt.savefig('./images/{}'.format(fig_name))


def plot_loss(training_loss: List, validation_loss: List, save_figure: bool = False, fig_name: str = None, show: bool = False):
    training_loss = np.array(training_loss)
    validation_loss = np.array(validation_loss)

    plt.figure(figsize=(5, 5), dpi=200)
    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    if show:
        plt.show()
    if save_figure:
        plt.savefig('./images/{}'.format(fig_name))
