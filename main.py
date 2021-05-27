import torch
import torch.nn as nn
from torch.optim import Adam
from models import RNNClassifier, LSTMClassifier, GRUClassifier
from data import Pendigits
import yaml
from typing import Dict
from torch.utils.data import DataLoader

# TODO: investigate why is 2 and not 8


def RNNClassifierTrainer(cfg: Dict):
    device = torch.device(cfg['system']['device'])
    lr = float(cfg['system']['lr'])
    num_epochs = int(cfg['rnn']['num_epochs'])
    batch_size = int(cfg['system']['batch_size'])

    model = RNNClassifier.RNNClassifier(cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ds = Pendigits.Pendigits('./data/pendigits.tra')
    dl = DataLoader(ds, num_workers=2, batch_size=batch_size)

    for epoch in range(1, num_epochs + 1):
        for idx, e in enumerate(dl):
            digits, labels = e
            digits = digits.float()
            digits = digits.to(device)
            labels = labels.to(device)

            out = model(digits)
            loss = criterion(out, labels)
            _, pred = torch.max(out, 1)

            num_correct = (pred == labels)
            num_correct = num_correct.sum()
            acc = num_correct.item() / len(labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx == len(dl) - 1:
                print('[RNN][{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch,
                    num_epochs, loss.item(), acc))

    # Save model
    torch.save(model.state_dict(), './trained_models/rnn.pt')


def LSTMClassifierTrainer(cfg: Dict):
    device = torch.device(cfg['system']['device'])
    lr = float(cfg['system']['lr'])
    num_epochs = int(cfg['lstm']['num_epochs'])
    batch_size = int(cfg['system']['batch_size'])

    model = LSTMClassifier.LSTMClassifier(cfg=cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ds = Pendigits.Pendigits('./data/pendigits.tra')
    dl = DataLoader(ds, num_workers=2, batch_size=batch_size)

    for epoch in range(1, num_epochs + 1):
        for idx, e in enumerate(dl):
            digits, labels = e
            digits = digits.float()
            digits = digits.to(device)
            labels = labels.to(device)

            out = model(digits)
            loss = criterion(out, labels)
            _, pred = torch.max(out, 1)

            num_correct = (pred == labels)
            num_correct = num_correct.sum()
            acc = num_correct.item() / len(labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx == len(dl) - 1:
                print('[LSTM][{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch,
                    num_epochs, loss.item(), acc))

    # Save model
    torch.save(model.state_dict(), './trained_models/lstm.pt')


def GRUClassifierTrainer(cfg: Dict):
    device = torch.device(cfg['system']['device'])
    lr = float(cfg['system']['lr'])
    num_epochs = int(cfg['gru']['num_epochs'])
    batch_size = int(cfg['system']['batch_size'])

    model = LSTMClassifier.LSTMClassifier(cfg=cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ds = Pendigits.Pendigits('./data/pendigits.tra')
    dl = DataLoader(ds, num_workers=2, batch_size=batch_size)

    for epoch in range(1, num_epochs + 1):
        for idx, e in enumerate(dl):
            digits, labels = e
            digits = digits.float()
            digits = digits.to(device)
            labels = labels.to(device)

            out = model(digits)
            loss = criterion(out, labels)
            _, pred = torch.max(out, 1)

            num_correct = (pred == labels)
            num_correct = num_correct.sum()
            acc = num_correct.item() / len(labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx == len(dl) - 1:
                print('[GRU][{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch,
                    num_epochs, loss.item(), acc))

    # Save model
    torch.save(model.state_dict(), './trained_models/gru.pt')


if __name__ == '__main__':
    with open('./config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    RNNClassifierTrainer(cfg=config)
    LSTMClassifierTrainer(cfg=config)
    GRUClassifierTrainer(cfg=config)
