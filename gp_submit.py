# Trains a basic random forest on both the fnc and the loadings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

root_path = '/home/nvme/Kaggle/trends-assessment-prediction'

loadings = pd.read_csv(f'{root_path}/loading.csv', index_col='Id')
fnc = pd.read_csv(f'{root_path}/fnc.csv', index_col='Id')
train_scores = pd.read_csv(f'{root_path}/train_scores.csv', index_col='Id')

sample_submission = pd.read_csv(f'{root_path}/sample_submission.csv', index_col='Id')

#x = pd.concat([loadings, fnc], axis=1)
x = loadings
x = x[~x.index.isin(train_scores.index)]

save_dict = pickle.load(open('gp_loadings.sklearn', 'rb'))
model = save_dict['model']
x_mean, x_std = save_dict['x_mean'], save_dict['x_std']

x = (x - x_mean) / (x_std + 1e-3)

# Why bother trying to format t hings when we can jsut sequentially fill it in
out = model.predict(x)
print(out)

out = out.flatten().tolist()
for i, o in enumerate(out):
    sample_submission.iloc[i] = o

print(sample_submission)

sample_submission.to_csv('gp_loadings_submission.csv')
