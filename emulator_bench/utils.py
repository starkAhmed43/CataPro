import os

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from scipy.stats import rankdata


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))


def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true), axis=-1)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true), axis=-1)


def pcc(y_true, y_pred):
    fsp = y_pred - np.mean(y_pred)
    fst = y_true - np.mean(y_true)
    dev_p = np.std(y_pred)
    dev_t = np.std(y_true)
    return np.mean(fsp * fst) / (dev_p * dev_t)


def scc(y_true, y_pred):
    x_rank = rankdata(y_pred)
    y_rank = rankdata(y_true)
    return pcc(y_rank, x_rank)


def r2_score(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - y_true_mean))
    return 1 - numerator / denominator


def evaluate(y_true, y_pred):
    return (
        pcc(y_true, y_pred),
        scc(y_true, y_pred),
        r2_score(y_true, y_pred),
        rmse(y_true, y_pred),
        mse(y_true, y_pred),
        mae(y_true, y_pred),
    )


def out_results(values, file_path):
    columns = ["valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_mse", "valid_mae", "valid_loss"]
    df = pd.DataFrame(values.reshape(1, -1), columns=columns)
    df.to_csv(file_path, float_format="%.5f", index=False)


def write_logfile(epoch, record_data, logfile):
    if epoch == 0 and os.path.exists(logfile):
        os.remove(logfile)

    values = np.array(record_data).reshape(epoch + 1, -1)
    columns = [
        "epoch",
        "train_pcc",
        "train_scc",
        "train_r2",
        "train_rmse",
        "train_mse",
        "train_mae",
        "train_loss",
        "valid_pcc",
        "valid_scc",
        "valid_r2",
        "valid_rmse",
        "valid_mse",
        "valid_mae",
        "valid_loss",
    ]
    df = pd.DataFrame(values, index=list(range(epoch + 1)), columns=columns)
    df.to_csv(logfile, float_format="%.4f")


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = 0
        self.count_epoch = 0
        self.stop = False
        self.is_bestmodel = False

    def check(self, epoch, cur_loss):
        if epoch == 0:
            self.min_loss = cur_loss
            self.count_epoch += 1
            self.is_bestmodel = True
        else:
            if cur_loss < self.min_loss - self.min_delta:
                self.min_loss = cur_loss
                self.count_epoch = 0
                self.is_bestmodel = True
            else:
                self.count_epoch += 1
                self.is_bestmodel = False

        if self.count_epoch == self.patience:
            self.stop = True

        return self.is_bestmodel, self.stop


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_true, y_pred):
        return th.sqrt(self.mse(y_true, y_pred) + self.eps)


rmse_loss = RMSELoss()


def run_a_training_epoch(model, data_loader, optimizer, device="cuda:0", skip_singleton_batch=False):
    model.train()

    total_loss = 0
    y_label = []
    y_pred = []

    for _, data in enumerate(data_loader):
        data = data.to(device)
        ezy_feats = data[:, :1024]
        sbt_feats = data[:, 1024:-1]
        label = data[:, -1]

        if skip_singleton_batch and len(label) == 1:
            continue

        pred = model(ezy_feats, sbt_feats)
        loss = rmse_loss(label, pred.ravel())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if th.cuda.is_available():
            th.cuda.empty_cache()

        with th.no_grad():
            total_loss += loss.cpu().detach().numpy()
            y_label.append(label.cpu().detach().numpy().ravel())
            y_pred.append(pred.cpu().detach().numpy().ravel())

    y_pred = np.concatenate(y_pred, axis=0)
    y_label = np.concatenate(y_label, axis=0)

    _pcc, _scc, _r2, _rmse, _mse, _mae = evaluate(y_label, y_pred)
    return np.array([_pcc, _scc, _r2, _rmse, _mse, _mae, total_loss / len(data_loader)])


def run_an_eval_epoch(model, data_loader, device="cuda:0"):
    model.eval()

    with th.no_grad():
        total_loss = 0
        y_label = []
        y_pred = []

        for _, data in enumerate(data_loader):
            data = data.to(device)
            ezy_feats = data[:, :1024]
            sbt_feats = data[:, 1024:-1]
            label = data[:, -1]

            pred = model(ezy_feats, sbt_feats)
            loss = rmse_loss(label, pred.ravel())

            total_loss += loss.cpu().detach().numpy()
            y_label.append(label.cpu().detach().numpy().ravel())
            y_pred.append(pred.cpu().detach().numpy().ravel())

        y_pred = np.concatenate(y_pred, axis=0)
        y_label = np.concatenate(y_label, axis=0)

        _pcc, _scc, _r2, _rmse, _mse, _mae = evaluate(y_label, y_pred)
        return y_pred, y_label, np.array([_pcc, _scc, _r2, _rmse, _mse, _mae, total_loss / len(data_loader)])
