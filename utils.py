import numpy as np
import torch

def numpy_metric(y_pred, y_true):
    return np.dot(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0), [0.3, 0.175, 0.175, 0.175, 0.175])

def torch_metric(y_pred, y_true):
    weights = torch.tensor([0.3, 0.175, 0.175, 0.175, 0.175]).to(device=y_pred.device)
    return torch.dot(torch.sum(torch.abs(y_true - y_pred), axis=0)/torch.sum(y_true, axis=0), weights)
