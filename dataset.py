import numpy as np
import pandas as pd
import nilearn as nl
import h5py

import torch
from torch.utils.data import Dataset

class NeuroimagingDataset(Dataset):
    def __init__(self, root_path, load_img=True, load_fnc=True, load_loadings=True, transforms_x=None, transforms_y=None, transforms_xy=None, train=True):
        self.root_path = root_path
        self.train = train
        self.load_img = load_img
        self.load_fnc = load_fnc
        self.load_loadings = load_loadings
        self.transforms_x = transforms_x
        self.transforms_y = transforms_y
        self.transforms_xy = transforms_xy

        self.loadings = pd.read_csv(f'{root_path}/loading.csv', index_col='Id')
        self.fnc = pd.read_csv(f'{root_path}/fnc.csv', index_col='Id')
        self.train_scores = pd.read_csv(f'{root_path}/train_scores.csv', index_col='Id')

        if train:
            self.ids = self.train_scores.index.tolist()
        else:
            self.ids = self.loadings.index[~self.loadings.isin(self.train_scores.index)].tolist()

        self.mask_fn = f'{root_path}/fMRI_mask.nii'
        self.mask_niimg = nl.image.load_img(self.mask_fn)


    def __getitem__(self, idx):
        loading = self.loadings.loc[self.ids[idx]]
        fnc = self.fnc.loc[self.ids[idx]]

        x = (loading, fnc)

        if self.load_img:
            if self.train:
                fn = f'{self.root_path}/fMRI_train/{self.ids[idx]}.mat'
            else:
                fn = f'{self.root_path}/fMRI_test/{self.ids[idx]}.mat'

            with h5py.File(fn, 'r') as f:
                niimg = f['SM_feature'][()]
            niimg = np.transpose(niimg, [3, 2, 1, 0]) # (x, y, z, feature_maps) from (feature_maps, z, y, x)
            niimg = nl.image.new_img_like(self.mask_niimg, niimg, affine=self.mask_niimg.affine, copy_header=True)

            x = (niimg,) + x
        if self.transforms_x:
            x = self.transforms_x(x)

        if self.train:
            y = self.train_scores.loc[self.ids[idx]]
            if self.transforms_y:
                y = self.transforms_y(y)
            if self.transforms_xy:
                x, y = self.transforms_xy(x, y)
            return x, y

        return x

    def __len__(self):
        return len(self.ids)
