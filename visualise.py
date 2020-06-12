import time

import numpy as np
import pandas as pd

import nilearn as nl
import nilearn.plotting as nlplt

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm

from dataset import NeuroimagingDataset, NormaliseImage

root_path = f'/home/nvme/Kaggle/trends-assessment-prediction'
train_scores = pd.read_csv(f'{root_path}/train_scores.csv', index_col='Id')
train_scores = train_scores.fillna(train_scores.mean())

train_set = NeuroimagingDataset(root_path, ids=train_scores.index.to_numpy(), train_scores=train_scores, transforms=NormaliseImage())
#train_loader = DataLoader(train_set, batch_size=16, shuffle=True,  pin_memory=True, num_workers=0)

#t0 = time.time()
#n = len(train_set)
#for i in tqdm(range(n)):
    #niimg, scores = train_set[i]
    #assert not torch.isnan(niimg).any()
    #assert not torch.isnan(scores).any()

#print(type(niimg), type(scores))

#t1 = time.time()
#print('Time per image', (t1 - t0) / n)

#train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  pin_memory=True, num_workers=0)

#t0 = time.time()
#n = 5
#for i, (x, y) in enumerate(train_loader):
    #if i >= n:
        #break
    #print(i, time.time() - t0)
#t1 = time.time()
#print('Time per batch', (t1 - t0) / n)


#train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  pin_memory=True, num_workers=0)

#t0 = time.time()
#n = 20
#for i, (x, y) in enumerate(train_loader):
    #if i >= n:
        #break
    #print(i, time.time() - t0)
#t1 = time.time()
#print('Time per batch', (t1 - t0) / n)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True,  pin_memory=True, num_workers=16)

t0 = time.time()
n = len(train_loader)
print(n)
for i, (x, y) in tqdm(enumerate(train_loader), total=n):
    if i >= n:
        break
    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()
t1 = time.time()
print('Time per batch', (t1 - t0) / n)
print(x.shape)

exit()

print('niimg shape', niimg.shape)

mask_niimg = nl.image.load_img(f'{root_path}/fMRI_mask.nii')
niimg = np.transpose(niimg.numpy(), [3, 2, 1, 0]) # (x, y, z, feature_maps) from (feature_maps, z, y, x)
niimg = nl.image.new_img_like(mask_niimg, niimg, affine=mask_niimg.affine, copy_header=True)
nlplt.plot_prob_atlas(niimg, view_type='filled_contours', draw_cross=False,title='All %d spatial maps' % niimg.shape[0], threshold='auto')


plt.show()
