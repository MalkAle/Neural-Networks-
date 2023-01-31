import numpy as np

from layer import Dense
from activation import ActivationSigmoid, ActivationTanh, ActivationReLu, ActivationLReLu, ActivationSoftmax

def load_param():
    try:
        with open('model.npy', 'rb') as model:
            param = np.load(model,allow_pickle=True) #input is a list of dictionaries for each layer of network
        print(f'Network loaded from file  as follows:\n{param}\n')
        return param
    except FileNotFoundError:
        print('\nERROR: file could no be found.\n')

def load_network(param):
    network = []
    for layer in param:
        layer_class_name = layer['layer_class_name']
        #print(f"layer_class_name is {layer['layer_class_name']}")
        layer_weight_shape = np.array(layer['weights']).shape
        if layer_weight_shape != ():
            layer_inst = eval(layer_class_name + '(layer_weight_shape[1], layer_weight_shape[0])')
        else:
            if layer_class_name == 'ActivationLReLu': #only Leaky ReLu needs the extra apha parameter
                layer_inst = eval(layer_class_name + "(layer['alpha'])") #this instatiates the layer classes dynamically from the input file
            else:    
                layer_inst = eval(layer_class_name + "()")
        network.append(layer_inst)
        layer_inst.weights = layer['weights']
        layer_inst.biases = layer['biases']
    
    return network

#Test
"""
param = load_param()
network = create_network(param)

print(network[0].weights)
"""