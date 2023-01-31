import numpy as np

from layer import *
from activation import *

class Network:
    def __init__(self, input_size: int, output_size: int, hidden_size: int, hidden_number: int):     
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_number = hidden_number


class Network_LReLuSoftmax(Network):
    def __init__(self, alpha: float, input_size, output_size, hidden_size, hidden_number):
        super().__init__(input_size, output_size, hidden_size, hidden_number)
        self.alpha = alpha
        network = []
        network.append(Dense(self.input_size,self.hidden_size))
        network.append(ActivationLReLu(self.alpha))
        if self.hidden_number > 1:
            for hidden_layer in range(self.hidden_number-1):
                network.append(Dense(self.hidden_size,self.hidden_size))
                network.append(ActivationLReLu(self.alpha))
        network.append(Dense(self.hidden_size,self.output_size))
        network.append(ActivationSoftmax())
        self.network = network

    def __repr__(self):
            return "LReLu-Softmax_Network"

class Network_Tanh(Network):    
    def __init__(self, alpha, input_size, output_size, hidden_size, hidden_number):
        super().__init__(input_size, output_size, hidden_size, hidden_number)
        network = []
        network.append(Dense(self.input_size,self.hidden_size))
        network.append(ActivationTanh())
        if self.hidden_number > 1:
            for hidden_layer in range(self.hidden_number-1):
                network.append(Dense(self.hidden_size,self.hidden_size))
                network.append(ActivationTanh())
        network.append(Dense(self.hidden_size,self.output_size))
        network.append(ActivationTanh())
        self.network = network

    def __repr__(self):
            return "Tanh_Network"

"""
network_1 = Network_LReLuSoftmax(alpha=0.05, input_size=2, output_size=3, hidden_size=5, hidden_number=1)
network_2 = Network_Tanh(input_size=2, output_size=3, hidden_size=5, hidden_number=1)
print(network_1.__dict__)
print(network_2.__dict__)
"""