import time

import numpy as np
import pandas as pd

import nilearn as nl
import nilearn.plotting as nlplt

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm

from dataset import NeuroimagingDataset

root_path = f'/home/nvme/Kaggle/trends-assessment-prediction'
train_scores = pd.read_csv(f'{root_path}/train_scores.csv', index_col='Id')
train_scores = train_scores.fillna(train_scores.mean())

train_set = NeuroimagingDataset(root_path, ids=train_scores.index.to_numpy(), train_scores=train_scores)
#train_loader = DataLoader(train_set, batch_size=16, shuffle=True,  pin_memory=True, num_workers=0)

niimg, scores = train_set[0]

niimg = np.transpose(niimg.numpy(), [3, 2, 1, 0]).astype(np.float32) # (x, y, z, feature_maps) from (feature_maps, z, y, x)

plt.figure()
plt.imshow(niimg[5, ..., 0])
plt.show()

mask_niimg = nl.image.load_img(f'{root_path}/fMRI_mask.nii')
niimg = nl.image.new_img_like(mask_niimg, niimg, affine=mask_niimg.affine, copy_header=True)
plt.figure()
nlplt.plot_prob_atlas(niimg, view_type='filled_contours', draw_cross=False,title='All %d spatial maps' % niimg.shape[0], threshold='auto')
plt.show()
