import numpy as np
import matplotlib.pyplot as plt

def create_data(samples, classes, data_kind):
    if data_kind == 'vertical':
        return vertical_data(samples,classes)
    elif data_kind == 'spiral':
        return spiral_data(samples, classes)

#https://cs231n.github.io/neural-networks-case-study/
def spiral_data(points, classes, factor1=4, factor2=1, plot=False):
    dimensions = 2 
    X = np.zeros((points*classes, dimensions))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.1, 1, points)  # radius
        t = np.linspace(class_number*factor1, (class_number+1)*factor1, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*factor2), r*np.cos(t*factor2)]
        y[ix] = class_number
    if plot:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()
    print(f'Spiral dataset created, shapes of X, Y are {X.shape},{y.shape}.\n')  
    return X, y

# Modified from:
# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def vertical_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    print(f'Vertical dataset created, shapes of X, Y are {X.shape},{y.shape}.\n')  
    return X, y

#Example
#X, y = spiral_data(300, 2, 3, plot=True)

#print(X)