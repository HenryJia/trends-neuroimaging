import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR

from tqdm import tqdm

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

loadings = pd.concat([loadings, (1.0 / 500.0) * fnc], axis=1)

x = loadings[loadings.index.isin(train_scores.index)].to_numpy().astype(np.float32)
y = train_scores.to_numpy().astype(np.float32)

kf = KFold(n_splits=5, shuffle=False)

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

p = Pool(14)
models = []
splits = list(kf.split(x))
for split in splits:
    x_train, y_train = x[split[0]], y[split[0]]

    model = [SVR(kernel='rbf', C=100, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True)]
    model = MultiModel(model)

    model.fit_async(p, x_train, y_train, blocking=True)
    models += [model]

out_val = []
for m, split in zip(models, splits):
    x_val, y_val = x[split[1]], y[split[1]]
    out_val += [m.predict(x_val)]

out_val = np.concatenate(out_val, axis=0)

model = [SVR(kernel='rbf', C=100, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True), SVR(kernel='rbf', C=10, verbose=True)]
model = MultiModel(model)

model.fit_async(p, x, y, blocking=True)

print('\nStddev')
print(y.std(axis=0))

print('\nValidation RMSE')
print(np.sqrt(np.mean((out_val - y) ** 2, axis=0)))

print('\nValidation Metric')
print(numpy_metric(out_val, y))

print('\nNumber of Support Vectors')
print([len(m.support_) for m in model.models])

sample_submission = pd.read_csv(f'{root_path}/sample_submission.csv', index_col='Id')

x_test = loadings[~loadings.index.isin(train_scores.index)].to_numpy().astype(np.float32)
#x_test = (x_test - x_test.mean(axis=0)) / (x_test.std(axis=0) + 1e-3)
out_test = model.predict(x_test).flatten()
print(out_test.shape)

for i, o in enumerate(out_test):
    sample_submission.iloc[i] = o

print(sample_submission)

sample_submission.to_csv('svm_loadings_submission.csv')

p.close()
