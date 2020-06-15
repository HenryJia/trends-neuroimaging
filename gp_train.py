# Trains a basic random forest on both the fnc and the loadings
import math
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.model_selection import train_test_split
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
x = x[x.index.isin(train_scores.index)]

x_train, x_val, y_train, y_val = train_test_split(x, train_scores, test_size=0.1)

x_mean, x_std = x_train.mean(), x_train.std()
x_train = (x_train - x_mean) / (x_std + 1e-3)
x_val = (x_val - x_mean) / (x_std + 1e-3)

y_mean, y_std = y_train.mean(), y_train.std()
y_train = (y_train - y_mean) / (y_std + 1e-3)
y_val = (y_val - y_mean) / (y_std + 1e-3)

model = GaussianProcessRegressor(kernel=0.9 * RationalQuadratic(0.1, 1.0) + 0.1 * RBF(1.0), normalize_y=True)
model.fit(x_train, y_train)

out_val = model.predict(x_val)
out_val = out_val * y_std.to_numpy() + y_mean.to_numpy()
y_val = y_val * y_std.to_numpy() + y_mean.to_numpy()

print('\nValidation stddev')
print(y_val.std())

print('\nValidation RMSE')
print(((out_val - y_val) ** 2).shape, type((out_val - y_val) ** 2))
print(np.sqrt(np.mean((out_val - y_val) ** 2)))

print('\nValidation Metric')
print(numpy_metric(out_val, y_val))

print('\nValidation R^2')
print(model.score(x_val, y_val))


pickle.dump({'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std, 'model': model}, open('gp_loadings.sklearn', 'wb'))
