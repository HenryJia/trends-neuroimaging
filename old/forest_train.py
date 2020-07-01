# Trains a basic random forest on both the fnc and the loadings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

x = pd.concat([loadings, fnc], axis=1)
x = x[x.index.isin(train_scores.index)]

x_train, x_val, y_train, y_val = train_test_split(x, train_scores, test_size=0.1)

model = RandomForestRegressor(n_estimators=200, n_jobs=15, verbose=1)
model.fit(x_train, y_train)

out_val = model.predict(x_val)

print('\nValidation stddev')
print(y_val.std())

print('\nValidation RMSE')
print(np.sqrt(np.mean((out_val - y_val) ** 2)))

print('\nValidation Metric')
print(numpy_metric(out_val, y_val))

print('\nValidation R^2')
print(model.score(x_val, y_val))


pickle.dump(model, open('forest_fnc_loadings.sklearn', 'wb'))

