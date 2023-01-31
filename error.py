import numpy as np


def mse(y, y_pred):    
    mse =  np.mean(np.power(y - y_pred, 2))
    mse_prime = 2 * (y_pred - y) / np.size(y)
    return mse, mse_prime

def cce(y, y_pred):
    cce = -np.sum(y * np.log(y_pred))
    cce_prime =  - y / y_pred
    return cce, cce_prime