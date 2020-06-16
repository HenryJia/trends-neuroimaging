# Trains a basic random forest on both the fnc and the loadings
import math
from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split, KFold
import pickle

from utils import numpy_metric

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

#x = pd.concat([loadings, fnc], axis=1)
x = loadings
x = x[x.index.isin(train_scores.index)].to_numpy()
y = train_scores.to_numpy()

def fit(split=None):
    global x, y
    if split:
        x_train, y_train = x[split[0]], y[split[0]]
        x_val, y_val = x[split[1]], y[split[1]]
    else:
        x_train = x
        y_train = y

    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_mean) / (x_std + 1e-3)

    if split:
        x_val = (x_val - x_mean) / (x_std + 1e-3)

    model = GaussianProcessRegressor(kernel=1.0 * RationalQuadratic(1.0, 1.0) + 1.0 * DotProduct(1.0) + 1.0 * WhiteKernel(1.0), normalize_y=True)
    model.fit(x_train, y_train)

    if split:
        out_val = model.predict(x_val)
        return np.sqrt(np.mean((out_val - y_val) ** 2, axis=0)), numpy_metric(out_val, y_val), model.score(x_val, y_val), model
    else:
        return model, x_mean, x_std

kf = KFold(n_splits=5, random_state=94103, shuffle=True)
splits = list(kf.split(x)) + [None]
p = Pool(len(splits) + 1)
results = p.map(fit, splits)
model, x_mean, x_std = results[-1]

rmse = np.mean([r[0] for r in results[:-1]], axis=0)
metric = np.mean([r[1] for r in results[:-1]], axis=0)
r2 = np.mean([r[2] for r in results[:-1]], axis=0)

print('\nValidation stddev')
print(y.std(axis=0))

print('\nValidation RMSE')
print(rmse)

print('\nValidation Metric')
print(metric)

print('\nValidation R^2')
print(r2)

print('\nPosterior Kernel')
print(model.kernel_)

pickle.dump({'x_mean': x_mean, 'x_std': x_std, 'model': model}, open('gp_loadings.sklearn', 'wb'))
