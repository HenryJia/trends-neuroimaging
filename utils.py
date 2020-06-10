import numpy as np

def metric(y_true, y_pred):
    return np.dot(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0), [0.3, 0.175, 0.175, 0.175, 0.175])
