import os
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR

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

from tqdm import tqdm

from utils import numpy_metric


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

class DownsampleBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv3d(in_dim, hidden_dim, 1),
                                   nn.ELU(),
                                   nn.Conv3d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
                                   nn.ELU())
        for i in range(n_hidden):
            self.convs.add_module(str(len(self.convs)), nn.Conv3d(hidden_dim, hidden_dim, 5, padding=2))
            self.convs.add_module(str(len(self.convs)),nn.ELU())
        self.convs.add_module(str(len(self.convs)), nn.Conv3d(hidden_dim, out_dim, 1))
        self.convs.add_module(str(len(self.convs)),nn.ELU())

        self.residual_conv = nn.Conv3d(in_dim, out_dim, 1)

    def forward(self, x):
        out = self.convs(x)
        return F.interpolate(self.residual_conv(x), out.shape[2:], mode='trilinear') + out

class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv3d(in_dim, hidden_dim, 1),
                                   nn.ELU())
        for i in range(n_hidden):
            self.convs.add_module(str(len(self.convs)), nn.Conv3d(hidden_dim, hidden_dim, 5, padding=2))
            self.convs.add_module(str(len(self.convs)),nn.ELU())

        self.convs.add_module(str(len(self.convs)), nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1))
        self.convs.add_module(str(len(self.convs)),nn.ELU())
        self.convs.add_module(str(len(self.convs)), nn.Conv3d(hidden_dim, out_dim, 1))
        self.convs.add_module(str(len(self.convs)),nn.ELU())

        self.residual_conv = nn.Conv3d(in_dim, out_dim, 1)

    def forward(self, x):
        out = self.convs(x)
        return F.interpolate(self.residual_conv(x), out.shape[2:], mode='trilinear') + out

class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv3d(in_dim, hidden_dim, 1),
                                   nn.ELU())
        for i in range(n_hidden):
            self.convs.add_module(str(len(self.convs)), nn.Conv3d(hidden_dim, hidden_dim, 5, padding=2))
            self.convs.add_module(str(len(self.convs)),nn.ELU())

        self.convs.add_module(str(len(self.convs)), nn.Conv3d(hidden_dim, out_dim, 1))
        self.convs.add_module(str(len(self.convs)),nn.ELU())

    def forward(self, x):
        return self.convs(x)

class DAE(LightningModule):
    def __init__(self, feature_dim=64, batch_size=16, lr=3e-4, noise=1, num_workers=8, q_size=2):
        super().__init__()
        self.noise = noise
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.q_size = q_size
        self.enc = nn.Sequential(DownsampleBlock(53, 4, feature_dim, 3), # (53, 52, 63, 53) -> (, 26, 32, 27)
                                 DownsampleBlock(feature_dim, 8, feature_dim, 3), # (, 26, 32, 27) -> (, 13, 16, 14)
                                 DownsampleBlock(feature_dim, 16, feature_dim, 3), # (, 13, 16, 14) -> (, 7, 8, 7)
                                 DownsampleBlock(feature_dim, 32, feature_dim, 3), # (, 7, 8, 7) -> (, 4, 4, 4)
                                 nn.Flatten(),
                                 nn.Linear(feature_dim * 4 ** 3, feature_dim),
                                 nn.ELU())

        self.dec_fc = nn.Sequential(nn.Linear(feature_dim, feature_dim * 4 ** 3),
                                    nn.ELU(),
                                    Reshape(feature_dim, 4, 4, 4))

        self.dec_convs = nn.Sequential(UpsampleBlock(feature_dim, 32, feature_dim, 3), # (, 4, 4, 4) -> (, 8, 8, 8)
                                       UpsampleBlock(feature_dim, 16, feature_dim, 3), # (, 8, 8, 8) -> (, 16, 16, 16)
                                       UpsampleBlock(feature_dim, 8, feature_dim, 3), # (, 16, 16, 16) -> (, 32, 32, 32)
                                       UpsampleBlock(feature_dim, 4, feature_dim, 3)) # (, 32, 32, 32) -> (53, 64, 64, 64)

        self.out_convs = nn.ModuleList([Block(feature_dim, 32, 53, 1),
                                        Block(feature_dim, 16, 53, 1),
                                        Block(feature_dim, 8, 53, 1),
                                        Block(feature_dim, 4, 53, 1),
                                        Block(feature_dim, 4, 53, 1)])

    def encode(self, niimg, noise=1):
        niimg = (niimg - niimg.mean(dim=(2, 3, 4), keepdims=True)) / (niimg.std(dim=(2, 3, 4), keepdims=True) + 1e-3)
        niimg = niimg + torch.randn_like(niimg) * noise

        embedding = self.enc(niimg)
        return embedding

    def forward(self, x, noise=1):
        embedding = self.encode(x, noise)

        reconstructions = [self.dec_fc(embedding)]
        for layer in self.dec_convs:
            reconstructions += [layer(reconstructions[-1])]

        reconstructions = [(x != 0).int() * F.interpolate(c(r), (52, 63, 53), mode='trilinear') for c, r in zip(self.out_convs, reconstructions)]
        return reconstructions

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

        self.train_set = NeuroimagingDataset(root_path, ids=x_train.index.to_numpy(), train_scores=y_train)
        self.val_set = NeuroimagingDataset(root_path, ids=x_val.index.to_numpy(), train_scores=y_val)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=self.num_workers)
        #return dataloader
        return AsynchronousLoader(dataloader, device=torch.device('cuda', 0), q_size=self.q_size, dtype=torch.float16 if args.precision == 16 else torch.float32)

    def val_dataloader(self):
        dataloader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,  pin_memory=False, num_workers=self.num_workers)
        #return dataloader
        return AsynchronousLoader(dataloader, device=torch.device('cuda', 0), q_size=self.q_size, dtype=torch.float16 if args.precision == 16 else torch.float32)

    def training_step(self, batch, batch_idx):
        niimg, y = batch

        out = self(niimg, noise=self.noise)
        loss = sum([F.mse_loss(o, niimg) for o in out])
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        niimg, y = batch
        out = self(niimg, noise=self.noise)
        loss = F.mse_loss(out[-1], niimg)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs}


class MultiModel:
    def __init__(self, models):
        self.models = models

    def fit_elemwise(self, m, x, y):
        m.fit(x, y)
        return m

    def fit_async(self, p, x, y, blocking=False):
        if blocking:
            self.models = p.starmap(self.fit_elemwise, [(m, x, y[:, i]) for i, m in enumerate(self.models)])
        else:
            self.fitted = p.starmap_async(self.fit_elemwise, [(m, x, y[:, i]) for i, m in enumerate(self.models)])

    def wait(self):
        self.models = self.fitted.get()

    def predict(self, x):
        return np.stack([m.predict(x) for m in self.models], axis=-1)


argument_parser = ArgumentParser(add_help=False)
argument_parser.add_argument('--precision', type=int, default=32, help='model precision')
argument_parser.add_argument('--batch_size', type=int, default=16, help='model precision')
argument_parser.add_argument('--epochs', type=int, default=20, help='model precision')
argument_parser.add_argument('--embedding_weight', type=float, default=0.002, help='model embedding weights')
argument_parser.add_argument('--noise', type=float, default=1, help='model noise')
args = argument_parser.parse_args()

logger = TensorBoardLogger("tb_logs", name="dae_fmri")
checkpoint_callback = ModelCheckpoint(filepath='./dae_fmri/{epoch:02d}-{val_loss:.2f}.ckpt', save_top_k=1, verbose=True, monitor='val_loss', mode='min', prefix='')
trainer = pl.Trainer(max_epochs=args.epochs, gpus=[0], precision=args.precision, logger=logger, checkpoint_callback=checkpoint_callback, use_amp=True if args.precision == 16 else False)

dae = DAE(batch_size=args.batch_size, noise=args.noise)
trainer.fit(dae)

root_path = '/home/nvme/Kaggle/trends-assessment-prediction'

loadings = pd.read_csv(f'{root_path}/loading.csv', index_col='Id')
fnc = pd.read_csv(f'{root_path}/fnc.csv', index_col='Id')
train_scores = pd.read_csv(f'{root_path}/train_scores.csv', index_col='Id')

print('loadings has {} NaNs or missing values'.format(loadings.isnull().sum().sum()))
print('fnc has {} NaNs or missing values'.format(fnc.isnull().sum().sum()))
print('train_scores has {} NaNs or missing values'.format(train_scores.isnull().sum().sum()))

print('Filling NaNs with means')
loadings = loadings.fillna(loadings.mean())
fnc = fnc.fillna(fnc.mean())
train_scores = train_scores.fillna(train_scores.mean())

loadings = pd.concat([loadings, (1 / 500.0) * fnc], axis=1)

x = loadings[loadings.index.isin(train_scores.index)]
y = train_scores

with torch.no_grad():
    dae = dae.cuda(0)
    dataset = NeuroimagingDataset(root_path, ids=x.index.to_numpy(), train_scores=y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=False, num_workers=8)
    dataloader = AsynchronousLoader(dataloader, device=torch.device('cuda', 0), q_size=2, dtype=torch.float16 if args.precision == 16 else torch.float32)
    embeddings = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        embeddings += [dae.encode(batch[0], noise=0)]
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.reshape(embeddings.shape[0], -1).cpu().numpy()

x = x.to_numpy().astype(np.float32)
y = y.to_numpy().astype(np.float32)

x = np.concatenate([x, args.embedding_weight * embeddings], axis=1)

kf = KFold(n_splits=5, shuffle=False)

p = Pool(14)
svrs = []
splits = list(kf.split(x))
for split in splits:
    x_train, y_train = x[split[0]], y[split[0]]

    svr = [SVR(kernel='rbf', C=100, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True)]
    svr = MultiModel(svr)

    svr.fit_async(p, x_train, y_train, blocking=True)
    svrs += [svr]

out_val = []
for m, split in zip(svrs, splits):
    x_val, y_val = x[split[1]], y[split[1]]
    out_val += [m.predict(x_val)]

out_val = np.concatenate(out_val, axis=0)

svr = [SVR(kernel='rbf', C=100, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True)]
svr = MultiModel(svr)

svr.fit_async(p, x, y, blocking=True)

print('\nStddev')
print(y.std(axis=0))

print('\nValidation RMSE')
print(np.sqrt(np.mean((out_val - y) ** 2, axis=0)))

print('\nValidation Metric')
print(numpy_metric(out_val, y))

print('\nNumber of Support Vectors')
print([len(m.support_) for m in svr.models])

sample_submission = pd.read_csv(f'{root_path}/sample_submission.csv', index_col='Id')

x_test = loadings[~loadings.index.isin(train_scores.index)]

with torch.no_grad():
    dataset = NeuroimagingDataset(root_path, ids=x_test.index.to_numpy())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=False, num_workers=8)
    dataloader = AsynchronousLoader(dataloader, device=torch.device('cuda', 0), q_size=2, dtype=torch.float16 if args.precision == 16 else torch.float32)
    embeddings = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        embeddings += [dae.encode(batch, noise=0)]
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.reshape(embeddings.shape[0], -1).cpu().numpy()

x_test = x_test.to_numpy().astype(np.float32)
x_test = np.concatenate([x_test, args.embedding_weight * embeddings], axis=1)
out_test = svr.predict(x_test).flatten()

print(out_test.shape)

for i, o in enumerate(out_test):
    sample_submission.iloc[i] = o

print(sample_submission)

sample_submission.to_csv('svm_loadings_submission.csv')

p.close()
