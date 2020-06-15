# Trains a basic neural network on both the fnc and the loadings
# This achieves score of 0.1649 on validation set with ensemble size of 10
from argparse import ArgumentParser
import copy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import torch_metric

class Model(LightningModule):
    def __init__(self, batch_size=128, lr=3e-4, input_dim=1404, n_networks=100):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        def make_network():
            return nn.Sequential(nn.Linear(input_dim, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 5))

        self.networks = nn.ModuleList([make_network() for i in range(n_networks)])

        self.initial_parameters = nn.ParameterList([copy.deepcopy(p) for p in self.parameters()])
        for p in self.initial_parameters:
            p.requires_grad = False

    def forward(self, x):
        return [n(x) for n in self.networks]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def prepare_data(self):
        root_path = '/home/nvme/Kaggle/trends-assessment-prediction'

        loadings = pd.read_csv(f'{root_path}/loading.csv', index_col='Id')
        fnc = pd.read_csv(f'{root_path}/fnc.csv', index_col='Id')
        train_scores = pd.read_csv(f'{root_path}/train_scores.csv', index_col='Id')

        #log.info('loadings has {} NaNs or missing values'.format(loadings.isnull().sum().sum()))
        #log.info('fnc has {} NaNs or missing values'.format(fnc.isnull().sum().sum()))
        #log.info('train_scores has {} NaNs or missing values'.format(train_scores.isnull().sum().sum()))

        #log.info('Filling NaNs with means')
        loadings = loadings.fillna(loadings.mean())
        fnc = fnc.fillna(fnc.mean())
        train_scores = train_scores.fillna(train_scores.mean())

        x = pd.concat([loadings, fnc], axis=1)
        x = x[x.index.isin(train_scores.index)]

        x_train, x_val, y_train, y_val = train_test_split(x, train_scores, test_size=0.1)
        x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
        x_val = torch.tensor(x_val.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

        mean, std = torch.mean(x_train, axis=0), torch.std(x_train, axis=0)
        x_train = (x_train - mean) / (std + 1e-3)
        x_val = (x_val - mean) / (std + 1e-3)

        self.train_set = TensorDataset(x_train, y_train)
        self.val_set = TensorDataset(x_val, y_val)

    def train_dataloader(self):
        #log.info('Training data loader called.')
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        #log.info('Validation data loader called.')
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.stack([torch_metric(o, y) for o in out], dim=0).sum()
        l2_reg = 1e-3 * torch.sum(torch.stack([torch.sum((p - p_old) ** 2) for p, p_old in zip(self.parameters(), self.initial_parameters)], dim=0))
        loss = loss + l2_reg
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = torch.stack(self(x), dim=0).mean(dim=0)
        loss = torch_metric(out, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs}

argument_parser = ArgumentParser(add_help=False)
argument_parser.add_argument('--gpu', type=int, default=0, help='which gpu')
argument_parser.add_argument('--precision', type=int, default=32, help='model precision')
args = argument_parser.parse_args()


logger = TensorBoardLogger("tb_logs", name="nn_fnc_loadings")
checkpoint_callback = ModelCheckpoint(filepath='./nn_fnc_loadings/', save_top_k=1, verbose=True, monitor='val_loss', mode='min', prefix='')
trainer = pl.Trainer(max_epochs=100, gpus=[args.gpu], distributed_backend='dp', precision=args.precision, logger=logger, checkpoint_callback=checkpoint_callback)
model = Model()
trainer.fit(model)
