import numpy as np
import random
#from data import spiral_data

def reshape(input):
    input = np.array(input)
    try:
        if len(input.shape) == 1:
            return np.reshape((input),(input.shape[0],1,1))
        else:
            return np.reshape((input),(input.shape[0],input.shape[1],1))
    except IndexError:
        print('\nERROR: not enough datapoints!')

def dummy(input):
    try:
        #print(f'input is\n{input}\n')
        uniques = np.unique(input)
        #print(f'{uniques} \n')
        dummy = np.zeros((len(input),len(uniques))).reshape(len(input),len(uniques),1)
        index = np.unique(uniques,return_index=True)[0][0]
        for point,vector in zip(input,dummy):
            index = int(np.unique(point,return_index=True)[0][0])  
            #print(f'current y point is\n{point}') 
            #print(f'current vector is\n{vector}')
            #print(f'current index is\n{index}')
            #print(vector[index][0])
            vector[index][0] = 1
        return dummy
    except IndexError:
        print('\nERROR: not enough datapoints!')

def train_test_split(X, Y, ratio):
    #merging X and Y so that when dividing into train and test X and Y values per datapoint ramain together
    data = np.column_stack((X,Y))
    #creating an array of all indices in dataset
    n = len(data)
    index = np.arange(0,n)
    rand_index = np.random.choice(index,n,replace=False)
    ind_train = rand_index[0 : int(ratio * n)]
    ind_test = rand_index[int(ratio * n) : n]
    X_train = data[ind_train][:, 0 : data.shape[1] - 1]
    Y_train = data[ind_train][:, -1]
    X_test = data[ind_test][:, 0 : data.shape[1] - 1]
    Y_test = data[ind_test][:, -1]
 
    #train = train
    test = data[ind_test]
    #Alternative solution
    #index_train = np.random.choice(index,size=int(ratio * len(data)),replace=False)
    #mask_X = np.isin(index,index_train)
    #reverted values from train are test
    #mask_Y = np.isin(index,index_train,invert=True)
    #train_X, train_Y = data[mask_X],data[mask_Y]

    return X_train, Y_train, X_test, Y_test


       
#Example dummy
"""
from spiral import spiral_data
X, Y = spiral_data(5, 2, 3)
X = reshape(X)
Y = reshape(Y)
print(f'shape of Y is \n{Y.shape}\n')

dummy_array = dummy(Y)
print(f'dummy is \n{dummy_array}')
print(f'shape of dummy is \n{dummy_array.shape}\n')
#print(np.append([1,1],[2,2],axis=1))
"""

#Example reshape
"""
from spiral import spiral_data
#X = [3, 5]
#y = [1]

#print(f'X before reshape is \n{X}')
#print(f'y before reshape is \n{y}')

X = reshape(X)
y = reshape(y)

#X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
#y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

#print(f'X after reshape is \n{X}\n')
#print(f'y after reshape is \n{y}\n')
"""

#Example train-test-slit
"""
X, Y = spiral_data(3, 2, 3)
X_train, Y_train, X_test, Y_test = train_test_split(X,Y,0.7)
print(f'shapes of X_train, Y_train, X_test, Y_test before are {X_train.shape}, {Y_train.shape}, {X_test.shape}, {Y_test.shape}\n')
"""