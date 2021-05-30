import torch
from typing import Dict, Tuple, List
from models import RNNClassifier, LSTMClassifier, GRUClassifier, CNNClassifier, TCNNClassifier
import torch.nn as nn
from data import Pendigits
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import Tensor
from evaluation_metrics import cm


def compute_validation_loss(model: nn.Module, cfg: Dict, criterion: nn.CrossEntropyLoss) -> Tensor:
    device = torch.device(cfg['system']['device'])
    model.eval()

    val_ds = Pendigits.Pendigits(file_path='./data/pendigits.tes')
    data, labels = val_ds.get_all()

    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    bs, seq_len, features = data.shape
    data = data.view(bs, features, seq_len)

    data = data.float()
    data = data.to(device)

    labels = labels.to(device)

    out = model(data)
    loss = criterion(out, labels)

    return loss


def RNNClassifierTrainer(cfg: Dict) -> Tuple[List, List, List]:
    device = torch.device(cfg['system']['device'])
    lr = float(cfg['system']['lr'])
    num_epochs = int(cfg['rnn']['num_epochs'])
    batch_size = int(cfg['system']['batch_size'])

    model = RNNClassifier.RNNClassifier(cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ds = Pendigits.Pendigits(file_path='./data/pendigits.tra')
    dl = DataLoader(ds, num_workers=2, batch_size=batch_size)

    train_loss = []
    validation_loss = []
    cms = []

    for epoch in range(1, num_epochs + 1):
        for idx, e in enumerate(dl):
            model.train()

            digits, labels = e
            bs, seq_len, features = digits.shape
            digits = digits.view(bs, features, seq_len)
            digits = digits.float()
            digits = digits.to(device)
            labels = labels.to(device)

            out = model(digits)
            loss = criterion(out, labels)
            train_loss.append(loss.cpu().data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss = compute_validation_loss(model=model, cfg=cfg, criterion=criterion).cpu().data.item()
            validation_loss.append(val_loss)

            if idx == len(dl) - 1:
                print('[RNN][{}/{}] Training loss: {:.6f} | Validation loss: {:.6f}'.format(
                    epoch,
                    num_epochs,
                    loss.cpu().data.item(),
                    val_loss))

                # Step evaluation with confusion matrix and accuracy score
                cms.append(cm(model=model, cfg=cfg))

    # Save model
    torch.save(model.state_dict(), './trained_models/rnn.pt')

    return train_loss, validation_loss, cms


def LSTMClassifierTrainer(cfg: Dict) -> Tuple[List, List, List]:
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

    train_loss = []
    validation_loss = []
    cms = []

    for epoch in range(1, num_epochs + 1):
        for idx, e in enumerate(dl):
            model.train()

            digits, labels = e
            bs, seq_len, features = digits.shape
            digits = digits.view(bs, features, seq_len)
            digits = digits.float()
            digits = digits.to(device)
            labels = labels.to(device)

            out = model(digits)
            loss = criterion(out, labels)
            train_loss.append(loss.cpu().data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss = compute_validation_loss(model=model, cfg=cfg, criterion=criterion).cpu().data.item()
            validation_loss.append(val_loss)

            if idx == len(dl) - 1:
                print('[RNN][{}/{}] Training loss: {:.6f} | Validation loss: {:.6f}'.format(
                    epoch,
                    num_epochs,
                    loss.cpu().data.item(),
                    val_loss))

                # Step evaluation with confusion matrix and accuracy score
                cms.append(cm(model=model, cfg=cfg))

    # Save model
    torch.save(model.state_dict(), './trained_models/lstm.pt')

    return train_loss, validation_loss, cms


def GRUClassifierTrainer(cfg: Dict) -> Tuple[List, List, List]:
    device = torch.device(cfg['system']['device'])
    lr = float(cfg['system']['lr'])
    num_epochs = int(cfg['gru']['num_epochs'])
    batch_size = int(cfg['system']['batch_size'])

    model = GRUClassifier.GRUClassifier(cfg=cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ds = Pendigits.Pendigits('./data/pendigits.tra')
    dl = DataLoader(ds, num_workers=2, batch_size=batch_size)

    train_loss = []
    validation_loss = []
    cms = []

    for epoch in range(1, num_epochs + 1):
        for idx, e in enumerate(dl):
            model.train()

            digits, labels = e
            bs, seq_len, features = digits.shape
            digits = digits.view(bs, features, seq_len)
            digits = digits.float()
            digits = digits.to(device)
            labels = labels.to(device)

            out = model(digits)
            loss = criterion(out, labels)
            train_loss.append(loss.cpu().data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss = compute_validation_loss(model=model, cfg=cfg, criterion=criterion).cpu().data.item()
            validation_loss.append(val_loss)

            if idx == len(dl) - 1:
                print('[RNN][{}/{}] Training loss: {:.6f} | Validation loss: {:.6f}'.format(
                    epoch,
                    num_epochs,
                    loss.cpu().data.item(),
                    val_loss))

                # Step evaluation with confusion matrix and accuracy score
                cms.append(cm(model=model, cfg=cfg))

    # Save model
    torch.save(model.state_dict(), './trained_models/gru.pt')

    return train_loss, validation_loss, cms


def CNNClassifierTrainer(cfg: Dict) -> Tuple[List, List, List]:
    device = torch.device(cfg['system']['device'])
    lr = float(cfg['system']['lr'])
    num_epochs = int(cfg['cnn']['num_epochs'])
    batch_size = int(cfg['system']['batch_size'])

    model = CNNClassifier.CNNClassifier(cfg=cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ds = Pendigits.Pendigits('./data/pendigits.tra')
    dl = DataLoader(ds, num_workers=2, batch_size=batch_size)

    train_loss = []
    validation_loss = []
    cms = []

    for epoch in range(1, num_epochs + 1):
        for idx, e in enumerate(dl):
            model.train()

            digits, labels = e
            bs, seq_len, features = digits.shape
            digits = digits.view(bs, features, seq_len)
            digits = digits.float()
            digits = digits.to(device)
            labels = labels.to(device)

            out = model(digits)
            loss = criterion(out, labels)
            train_loss.append(loss.cpu().data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss = compute_validation_loss(model=model, cfg=cfg, criterion=criterion).cpu().data.item()
            validation_loss.append(val_loss)

            if idx == len(dl) - 1:
                print('[CNN][{}/{}] Training loss: {:.6f} | Validation loss: {:.6f}'.format(
                    epoch,
                    num_epochs,
                    loss.cpu().data.item(),
                    val_loss))

                # Step evaluation with confusion matrix and accuracy score
                cms.append(cm(model=model, cfg=cfg))

    # Save model
    torch.save(model.state_dict(), './trained_models/cnn.pt')

    return train_loss, validation_loss, cms


def TCNClassifierTrainer(cfg: Dict) -> Tuple[List, List, List]:
    device = torch.device(cfg['system']['device'])
    lr = float(cfg['system']['lr'])
    num_epochs = int(cfg['tcn']['num_epochs'])
    batch_size = int(cfg['system']['batch_size'])

    model = TCNNClassifier.TCNClassifier(cfg=cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ds = Pendigits.Pendigits('./data/pendigits.tra')
    dl = DataLoader(ds, num_workers=2, batch_size=batch_size)

    train_loss = []
    validation_loss = []
    cms = []

    for epoch in range(1, num_epochs + 1):
        for idx, e in enumerate(dl):
            model.train()

            digits, labels = e
            bs, seq_len, features = digits.shape
            digits = digits.view(bs, features, seq_len)
            digits = digits.float()
            digits = digits.to(device)
            labels = labels.to(device)

            out = model(digits)
            loss = criterion(out, labels)
            train_loss.append(loss.cpu().data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss = compute_validation_loss(model=model, cfg=cfg, criterion=criterion).cpu().data.item()
            validation_loss.append(val_loss)

            if idx == len(dl) - 1:
                print('[TCN][{}/{}] Training loss: {:.6f} | Validation loss: {:.6f}'.format(
                    epoch,
                    num_epochs,
                    loss.cpu().data.item(),
                    val_loss))

                # Step evaluation with confusion matrix and accuracy score
                cms.append(cm(model=model, cfg=cfg))

    # Save model
    torch.save(model.state_dict(), './trained_models/tcn.pt')

    return train_loss, validation_loss, cms
