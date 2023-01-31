import numpy as np


def process_forward(x, network):
    layer_output = x
    for layer in network:
        layer_output = layer.forward(layer_output)
    y_pred = np.clip(layer_output, 1e-7, 1-1e-7)#network output is limited to values from "a tiny bit above 0" and "a tiny bit below 1"
    #print(f'output of the network for this datapoint is\n {network_output}\n')
    pred_res = np.argmax(y_pred) #appends the index of max value of network output which is the predicted class
    pred_prob = max(np.around(y_pred,decimals=2)) #appends the probability of class, rounds value 
    return y_pred, pred_res, pred_prob

def process_backward(y_pred, y_train, network, learning_rate, error_func): 
    error, error_prime = error_func(y_train, y_pred)
    error_gradientY = error_prime  
    layer_input = error_gradientY
    for layer in reversed(network):
        layer_input = layer.backward(layer_input, learning_rate) 
    return(error)
        