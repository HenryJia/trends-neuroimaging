# Trains a basic neural network on both the fnc and the loadings
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

root_path = '/home/nvme/Kaggle/trends-assessment-prediction'

loadings = pd.read_csv(f'{root_path}/loading.csv', index_col='Id')
fnc = pd.read_csv(f'{root_path}/fnc.csv', index_col='Id')
train_scores = pd.read_csv(f'{root_path}/train_scores.csv', index_col='Id')

sample_submission = pd.read_csv(f'{root_path}/sample_submission.csv', index_col='Id')

x = pd.concat([loadings, fnc], axis=1)
x = x[~x.index.isin(train_scores.index)]
x = (x - x.mean()) / (x.std() + 1e-3)

model_path = f'./nn_fnc_loadings/_ckpt_epoch_97.ckpt'

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
        return torch.mean(torch.stack([n(x) for n in self.networks], dim=0), dim=0)

model = Model.load_from_checkpoint(model_path).cuda(0)

# Why bother trying to format t hings when we can jsut sequentially fill it in
out = model(torch.tensor(x.to_numpy(), dtype=torch.float32).cuda(0)).flatten().tolist()
for i, o in enumerate(out):
    sample_submission.iloc[i] = o

print(sample_submission)

sample_submission.to_csv('nn_fnc_loadings_submission.csv')
