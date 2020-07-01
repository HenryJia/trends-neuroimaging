from threading import Thread
from queue import Queue

import numpy as np
import pandas as pd
import nilearn as nl
import h5py

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NeuroimagingDataset(Dataset):
    def __init__(self, root_path, ids, loadings=None, fnc=None, train_scores=None, transforms=None):
        self.root_path = root_path
        self.ids = ids
        self.loadings = loadings
        self.fnc = fnc
        self.train_scores = train_scores
        self.transforms = transforms

        #self.mask_fn = f'{root_path}/fMRI_mask.nii'
        #self.mask_niimg = nl.image.load_img(self.mask_fn) # Don't need this for now


    def __getitem__(self, idx):
        if self.train_scores is not None:
            fn = f'{self.root_path}/fMRI_train/{self.ids[idx]}.mat'
        else:
            fn = f'{self.root_path}/fMRI_test/{self.ids[idx]}.mat'

        with h5py.File(fn, 'r') as f:
            niimg = f.get('SM_feature')[()]
        if self.transforms is not None:
            niimg = self.transforms(niimg)
        niimg = torch.tensor(niimg, dtype=torch.float16)

        # PyTorch uses the convention of (feature_maps, depth, height, width), so no need for this
        #niimg = np.transpose(niimg, [0, 3, 2, 1])  # (feature_maps, x, y, z) from (feature_maps, z, y, x)

        # We only need to do this for visualisations for now
        #niimg = np.transpose(niimg, [3, 2, 1, 0]) # (x, y, z, feature_maps) from (feature_maps, z, y, x)
        #niimg = nl.image.new_img_like(self.mask_niimg, niimg, affine=self.mask_niimg.affine, copy_header=True)

        x = (niimg,)
        if self.loadings is not None:
            x += (torch.tensor(self.loadings.loc[self.ids[idx]].to_numpy(), dtype=torch.float16),)
        if self.fnc is not None:
            x += (torch.tensor(self.fnc.loc[self.ids[idx]].to_numpy(), dtype=torch.float16),)
        if len(x) == 1:
            x = x[0]

        if self.train_scores is not None:
            y = torch.tensor(self.train_scores.loc[self.ids[idx]].to_numpy(), dtype=torch.float16)
            return x, y

        return x


    def __len__(self):
        return len(self.ids)



class AsynchronousLoader(object):
    """
    Class for asynchronously loading from CPU memory to device memory with DataLoader
    Note that this only works for single GPU training, multiGPU uses PyTorch's DataParallel or
    DistributedDataParallel which uses its own code for transferring data across GPUs. This could just
    break or make things slower with DataParallel or DistributedDataParallel
    Parameters
    ----------
    data: PyTorch Dataset or PyTorch DataLoader
        The PyTorch Dataset or DataLoader we're using to load.
    device: PyTorch Device
        The PyTorch device we are loading to
    q_size: Integer
        Size of the queue used to store the data loaded to the device
    num_batches: Integer or None
        Number of batches to load.
        This must be set if the dataloader doesn't have a finite __len__
        It will also override DataLoader.__len__ if set and DataLoader has a __len__
        Otherwise can be left as None
    **kwargs:
        Any additional arguments to pass to the dataloader if we're constructing one here
    """

    def __init__(self, data, device=torch.device('cuda', 0), q_size=10, num_batches=None, dtype=None, **kwargs):
        if isinstance(data, torch.utils.data.DataLoader):
            self.dataloader = data
        else:
            self.dataloader = DataLoader(data, **kwargs)

        if num_batches is not None:
            self.num_batches = num_batches
        elif hasattr(self.dataloader, '__len__'):
            self.num_batches = len(self.dataloader)
        else:
            raise Exception("num_batches must be specified or data must have finite __len__")

        self.device = device
        self.q_size = q_size
        self.dtype = dtype

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.q_size)

        self.idx = 0

    def load_loop(self):  # The loop that will load into the queue in the background
        for i, sample in enumerate(self.dataloader):
            self.queue.put(self.load_instance(sample))
            if i == len(self):
                break

    # Recursive loading for each instance based on torch.utils.data.default_collate
    def load_instance(self, sample):
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                # Can only do asynchronous transfer if we use pin_memory
                if not sample.is_pinned():
                    sample = sample.pin_memory()
                if self.dtype:
                    return sample.to(self.device, non_blocking=True, dtype=self.dtype)
                else:
                    return sample.to(self.device, non_blocking=True)
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        # We don't want to run the thread more than once
        # Start a new thread if we are at the beginning of a new epoch, and our current worker is dead
        if (not hasattr(self, 'worker') or not self.worker.is_alive()) and self.queue.empty() and self.idx == 0:
            self.worker = Thread(target=self.load_loop)
            self.worker.daemon = True
            self.worker.start()
        return self

    def __next__(self):
        # If we've reached the number of batches to return
        # or the queue is empty and the worker is dead then exit
        done = not self.worker.is_alive() and self.queue.empty()
        done = done or self.idx >= len(self)
        if done:
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        else:  # Otherwise return the next batch
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def __len__(self):
        return self.num_batches
