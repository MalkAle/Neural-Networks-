import numpy as np
from time import time

from process import process_forward, process_backward

class Training:
    def __init__(self, error_func: str, learning_rate: float, epochs: int):
        self.error_func = error_func
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, network, X_train_r, Y_train_rd):
            print('Initiating Training.')
            start = time()
            self.pred_res = [] #predicted class
            self.pred_prob = []
            self.error_epoch_tracking = np.array([]) #error per epoch
            self.error_point_tracking = np.array([])
            for epoch in range(self.epochs):
                error_sum = 0
                for x_train, y_train in zip(X_train_r, Y_train_rd):  
                    y_pred, pred_res, pred_prob = process_forward(x_train, network)
                    error = process_backward(y_pred, y_train, network, self.learning_rate, self.error_func)
                    self.pred_res.append(pred_res)
                    self.pred_prob.append(pred_prob)
                    self.error_point_tracking = np.append(self.error_point_tracking, error)
                    error_sum += error
                self.error_epoch = error_sum / len(X_train_r)
                self.error_epoch_tracking = np.append(self.error_epoch_tracking, self.error_epoch) 
                print(f'Iteration no {epoch}, error for this epoch is {self.error_epoch}')
            end = time()
            print(f'Training completed, elapsed time is {end - start}\n') 



#Plotting loss functions
"""
def bce(y, y_pred):
    return  -np.sum(y * np.log(y_pred + 10**-100))

y_pred = np.arange(1,1,0.1)
print(y_pred.shape)
y = np.zeros((len(y_pred)))
print(y.shape)
delta = y_pred - y
print(delta.shape)
error_bce = bce(1, 0.0001)
print(error_bce)
#import matplotlib.pyplot as plt
#plt.plot(delta, error_bce)
#plt.show()
"""