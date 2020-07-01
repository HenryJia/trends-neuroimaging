import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

import torch
from torch import optim

import pyro
import pyro.contrib.gp as gp
from pyro.nn.module import PyroParam
import pyro.distributions as dist

from tqdm import tqdm

from utils import torch_metric

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

x = torch.tensor(loadings[loadings.index.isin(train_scores.index)].to_numpy()).cuda(0).float()
y = torch.tensor(train_scores.to_numpy()).cuda(0).float()

def fit(split=None):
    global x, y
    if split:
        x_train, y_train = x[split[0]], y[split[0]]
        x_val, y_val = x[split[1]], y[split[1]]
    else:
        x_train = x
        y_train = y

    x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)
    y_mean, y_std = y_train.mean(dim=0), y_train.std(dim=0)
    x_train = (x_train - x_mean) / (x_std + 1e-3)
    y_train = (y_train - y_mean) / (y_std + 1e-3)

    if split:
        x_val = (x_val - x_mean) / (x_std + 1e-3)

    k0 = gp.kernels.RationalQuadratic(input_dim=x.shape[1], variance=torch.tensor(1.), lengthscale=torch.tensor(1.))
    k1 = gp.kernels.WhiteNoise(input_dim=x.shape[1], variance=torch.tensor(1.))
    gpr = gp.models.GPRegression(x_train, y_train.T, kernel=gp.kernels.Sum(k0, k1), noise=torch.tensor(1.)).cuda(0)

    optimizer = optim.Adam(gpr.parameters(), lr=0.005)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    num_steps = 500
    pb = tqdm(total=num_steps)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()

        pb.set_postfix(loss=loss.item())
        pb.update(1)
    pb.close()

    if split:
        out, cov = gpr(x_val, full_cov=False, noiseless=True)
        return gpr, out.T * y_std + y_mean
    else:
        return gpr

kf = KFold(n_splits=5, shuffle=False)
out_val = []
for split in kf.split(x):
    model, out = fit(split)
    out_val += [out]

model = fit()

out_val = torch.cat(out_val, dim=0)

print('\nStddev')
print(y.std(axis=0))

print('\nValidation RMSE')
print(torch.sqrt(torch.mean((out_val - y) ** 2, dim=0)))

print('\nValidation Metric')
print(torch_metric(out_val, y))

print('\nPosterior Kernel')
print(model.kernel)

sample_submission = pd.read_csv(f'{root_path}/sample_submission.csv', index_col='Id')

x_test = torch.tensor(loadings[~loadings.index.isin(train_scores.index)].to_numpy()).cuda(0).float()
x_test = (x_test - x_test.mean(dim=0)) / (x_test.std(dim=0) + 1e-3)
out_test, cov = model(x_test)
out_test = out_test.T * y.std(dim=0) + y.mean(dim=0)

out_test = out_test.flatten().cpu().detach().numpy().tolist()
for i, o in enumerate(out_test):
    sample_submission.iloc[i] = o

print(sample_submission)

sample_submission.to_csv('gp_pyro_submission.csv')
