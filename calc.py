import numpy as np

from training import Training
from testing import Testing
from activation import ActivationTanh, ActivationSigmoid, ActivationReLu, ActivationLReLu, ActivationSoftmax
from networks import Network_Tanh, Network_LReLuSoftmax
from error import mse, cce

#The goal of this module is to create predefined network architectures and the appropriate loss function for training
# ensure that binary crossentropy loss function is used with softmax activation. The are 2 variants:
#- CalcSoftmax: Hidden layer activation function Leaky Relu, output layer Softmax, loss function is Binary-Cross-Entropy
#- CalcTanh: all layers Tanh, loss function is Mean Square Error  

def create_calc(calc_kind, hidden_size, hidden_number, alpha):
    if calc_kind == 'CalcSoftmax':
        return CalcSoftmax(hidden_size, hidden_number, alpha)
    elif calc_kind == 'CalcTanh':
        return CalcTanh(hidden_size, hidden_number)

class Calc:
    def __init__(self, hidden_size, hidden_number, alpha, network_type, error_func):
        self.hidden_size = hidden_size
        self.hidden_number = hidden_number
        self.alpha = alpha
        self.network_type = network_type
        self.error_func = error_func

    def create_network(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.network = self.network_type(self.alpha, input_size, output_size, self.hidden_size, self.hidden_number)
        print(f'\nNetwork created as follows:\n{self.network.__dict__}\n')
    
    def create_training(self, learning_rate, epochs):
        self.training = Training(self.error_func, learning_rate, epochs)
        print(f'Training created as follows:\n{self.training.__dict__}\n')
    
    def calc_train(self, X_train_r, Y_train_rd):
        self.training.train(self.network.network, X_train_r, Y_train_rd)

    def save(self):
        param = []
        for layer in self.network.network:
            #print(f'layer is {layer}')
            #print(f'alpha is {layer.alpha}')
            param.append({'layer_class_name': layer.__class__.__name__, 'alpha': layer.alpha, 'weights' : layer.weights, 'biases': layer.biases})
        with open('model.npy', 'wb') as model:
            np.save(model, param)
        print('Network saved successfully.\n')

class CalcTanh(Calc):
    def __init__(self, hidden_size, hidden_number):
        network_type = Network_Tanh
        alpha = None
        error_func = mse
        super().__init__(hidden_size, hidden_number, alpha, network_type, error_func)

class CalcSoftmax(Calc):
    def __init__(self, hidden_size, hidden_number, alpha):
        network_type = Network_LReLuSoftmax
        error_func = cce
        super().__init__(hidden_size, hidden_number, alpha, network_type, error_func)
