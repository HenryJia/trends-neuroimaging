import time

import numpy as np

import nilearn.plotting as nlplt

import matplotlib.pyplot as plt

from dataset import NeuroimagingDataset

train_set = NeuroimagingDataset('/home/nvme/Kaggle/trends-assessment-prediction', train=True)

t0 = time.time()
for i in range(10):
    (niimg, loadings, fnc), scores = train_set[i]
    print('niimg {} shape'.format(i), niimg.shape)

print('loadings fnc shape', loadings.shape, fnc.shape)

t1 = time.time()
print('Time per image', (t1 - t0) / 10)

nlplt.plot_prob_atlas(niimg, view_type='filled_contours', draw_cross=False,title='All %d spatial maps' % niimg.shape[0], threshold='auto')
plt.show()
