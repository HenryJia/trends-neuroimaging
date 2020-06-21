# Trains a basic neural network on both the fnc and the loadings
# This achieves score of 0.1649 on validation set with ensemble size of 10
import math
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import torch_metric
from dataset import NeuroimagingDataset, AsynchronousLoader

class Model(LightningModule):
    def __init__(self, feature_dim=128, batch_size=16, lr=3e-4, num_workers=8, q_size=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.q_size = q_size
        self.conv = nn.Sequential(nn.Conv3d(53, feature_dim // 2, 5, stride=2, padding=2), # (53, 52, 63, 53) -> (128, 26, 32, 27)
                                  nn.LeakyReLU(),
                                  nn.Conv3d(feature_dim // 2, feature_dim, 5, stride=2, padding=2), # (128, 26, 32, 27) -> (256, 13, 16, 14)
                                  nn.LeakyReLU(),
                                  nn.Conv3d(feature_dim, feature_dim, 5, stride=2, padding=2), # (256, 13, 16, 14) -> (512, 7, 8, 7)
                                  nn.LeakyReLU(),
                                  nn.Conv3d(feature_dim, feature_dim, 5, stride=2, padding=2), # (512, 7, 8, 7) -> (512, 4, 4, 4)
                                  nn.AvgPool3d(4),
                                  nn.Flatten())

        #self.fc_fnc_loadings = nn.Sequential(nn.Linear(1404, feature_dim),
                                             #nn.ELU())

        self.fc_combined = nn.Sequential(nn.Linear(feature_dim, 256),
                                         nn.ELU(),
                                         nn.Linear(256, 256),
                                         nn.ELU(),
                                         nn.Linear(256, 5))


    def forward(self, x, use_noise=False):
        niimg = (x - x.mean(dim=(2, 3, 4), keepdims=True)) / (x.std(dim=(2, 3, 4), keepdims=True) + 1e-3)
        niim = (niimg > 1).to(dtype=x.dtype)
        features_img = self.conv(niimg)

        #mean, log_var = features_img.chunk(2, dim=1)
        #if use_noise:
            #features_img = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
        #else:
            #features_img = mean

        #features_img, attention = torch.chunk(features_img, 2, dim=1)
        #features_img = features_img.reshape(-1, self.feature_dim // 2, 4 ** 3)
        #attention = attention.reshape(-1, self.feature_dim // 2, 4 ** 3)
        #features_img = torch.sum(features_img * torch.softmax(attention, dim=-1), dim=-1)
        #features_fnc_loadings = self.fc_fnc_loadings(fnc_loadings)

        #attention = self.fc_fnc_loadings(fnc_loadings).reshape(-1, 4 ** 3, self.attention_heads)
        #attention = F.softmax(attention / math.sqrt(4 ** 3))

        #out = self.fc_combined(features_fnc_loadings + features_img)
        out = self.fc_combined(features_img)
        if use_noise:
            return out, (mean, log_var)
        else:
            return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def prepare_data(self):
        root_path = '/home/nvme/Kaggle/trends-assessment-prediction'

        loadings = pd.read_csv(f'{root_path}/loading.csv', index_col='Id')
        fnc = pd.read_csv(f'{root_path}/fnc.csv', index_col='Id')
        train_scores = pd.read_csv(f'{root_path}/train_scores.csv', index_col='Id')

        loadings = loadings.fillna(loadings.mean())
        fnc = fnc.fillna(fnc.mean())
        train_scores = train_scores.fillna(train_scores.mean())

        x = pd.concat([loadings, fnc], axis=1)
        x = x[x.index.isin(train_scores.index)]

        x_train, x_val, y_train, y_val = train_test_split(x, train_scores, test_size=0.1)
        x_train = (x_train - x_train.mean()) / (x_train.std() + 1e-3)
        x_val = (x_val - x_train.mean()) / (x_train.std() + 1e-3)

        self.train_set = NeuroimagingDataset(root_path, ids=x_train.index.to_numpy(), fnc=x_train, train_scores=y_train)
        self.val_set = NeuroimagingDataset(root_path, ids=x_val.index.to_numpy(), fnc=x_val, train_scores=y_val)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=self.num_workers)
        #return dataloader
        return AsynchronousLoader(dataloader, device=torch.device('cuda', 0), q_size=self.q_size, dtype=torch.float16 if args.precision == 16 else torch.float32)

    def val_dataloader(self):
        dataloader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True,  pin_memory=False, num_workers=self.num_workers)
        #return dataloader
        return AsynchronousLoader(dataloader, device=torch.device('cuda', 0), q_size=self.q_size, dtype=torch.float16 if args.precision == 16 else torch.float32)

    def training_step(self, batch, batch_idx):
        (niimg, fnc_loadings), y = batch

        #out, (mean, log_var) = self(niimg, use_noise=True)
        #kl = 0.5 * torch.mean(torch.sum(mean ** 2 + log_var.exp() - log_var - 1, dim=1))
        #metric = torch_metric(out, y)

        #loss = kl + metric
        #logs = {'train_loss': loss, 'train_metric': metric, 'train_kl': kl}
        #return {'loss': loss, 'progress_bar': {'metric': metric, 'kl': kl, 'mean':mean.mean(), 'var': log_var.exp().mean()}, 'log': logs}

        out = self(niimg, use_noise=False)
        loss = torch_metric(out, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        (niimg, fnc_loadings), y = batch
        out = self(niimg, use_noise=False)
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

logger = TensorBoardLogger("tb_logs", name="nn_fmri")
checkpoint_callback = ModelCheckpoint(filepath='./nn_fmri/{epoch:02d}-{val_loss:.2f}.ckpt', save_top_k=1, verbose=True, monitor='val_loss', mode='min', prefix='')
trainer = pl.Trainer(max_epochs=100, gpus=[args.gpu], precision=args.precision, logger=logger, checkpoint_callback=checkpoint_callback, use_amp=True if args.precision == 16 else False)
model = Model()
trainer.fit(model)
