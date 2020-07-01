import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, ParameterGrid, GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.svm import SVR

from tqdm import tqdm


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

x = loadings[loadings.index.isin(train_scores.index)].astype(np.float32)
y = train_scores.astype(np.float32)

def numpy_scorer(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0)

scorer = make_scorer(numpy_scorer, greater_is_better=False)
param_grid = {'C': [1, 16, 64, 128], 'epsilon': [0.1, 1, 4, 8]}

models = []
for c in y.columns:
    model = GridSearchCV(SVR(kernel='rbf', cache_size=4000), param_grid=param_grid, scoring=scorer, cv=5, refit=True, verbose=1, n_jobs=14)
    model.fit(x, y[c])

    #print('GridSearchCV results for column', c)
    #print(model.cv_results_)

    print('GridSearchCV best params for column', c)
    print(model.best_score_)

    print('GridSearchCV best metric for column', c)
    print(model.best_params_)

    models += [model]


print('Overall Validation Metric')
print(np.dot([m.best_score_ for m in models], [0.3, 0.175, 0.175, 0.175, 0.175]))

sample_submission = pd.read_csv(f'{root_path}/sample_submission.csv', index_col='Id')

x_test = loadings[~loadings.index.isin(train_scores.index)].astype(np.float32)

out_test = np.stack([m.predict(x_test) for m in models], axis=1)
print(out_test.shape)
out_test = out_test.flatten().tolist()

for i, o in enumerate(out_test):
    sample_submission.iloc[i] = o

print(sample_submission)

sample_submission.to_csv('svm_submission.csv')
