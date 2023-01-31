import numpy as np
import matplotlib.pyplot as plt
import warnings

from layer import Layer



class Activation(Layer):
    #"activation" and "activation_prime" are functions
    def __init__(self, activation, activation_prime):
        #print('Initializing Activation layer\n')
        self.activation = activation
        self.activation_prime = activation_prime
        super().__init__()

    #Forward method simply applies the activation function to the input f(X)
    def forward(self, input):
        #print('Forward method in Activaiton called')
        self.input = input
        #print(f'Output of forward method in activation is \n{self.activation(self.input)}\n')
        return self.activation(self.input)

    #Backward method calculates the input gradient to feed in the previous layer
    #through Hadamard Product of output_gradient and activaton_prime
    def backward(self, output_gradient, learning_rate):
        #print('backward method in Activation called')
        self.activation_prime_output = self.activation_prime(self.input)
        input_gradient = np.multiply(output_gradient, self.activation_prime(self.input))
        #print(f'output of backward method in Activation is\n{input_gradient}\n')
        return input_gradient


class ActivationTanh(Activation):
    def __init__(self):
        #lambda fuction is used to pass functions like arguments
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        #Initializing the parent class with the attributes tanh and tanh_prime as functions
        #Calling super method to initialize the parent class Activation to use it's forward and backward functions
        super().__init__(tanh, tanh_prime)

    def __repr__(self):
        return "ActivationTanh"


class ActivationReLu(Activation):
    def __init__(self):
        def relu(x):
            return (x > np.zeros((len(x), 1))) * x

        def relu_prime(x):
            return (x > np.zeros((len(x), 1))) * 1

        super().__init__(relu, relu_prime)

    def __repr__(self):
        return "ActivationReLu"


class ActivationLReLu(Activation):
    def __init__(self, alpha):
        def lrelu(x):
            return np.where(x > 0, x, x * alpha) 

        def lrelu_prime(x):
            return np.where(x > 0, 1, alpha)

        super().__init__(lrelu, lrelu_prime)
        self.alpha = alpha #updating alpha after the init method in parend class layer has been set to None

    def __repr__(self):
        return "ActivationLReLu"

class ActivationSigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2
        super().__init__(sigmoid, sigmoid_prime)

    def __repr__(self):
        return "ActivationSigmoid"


class ActivationSoftmax(Layer):
    def __init__(self):
        super().__init__()

        #Forward method simply applies the activation function to the input f(X)

    def __repr__(self):
        return "ActivationSoftmax"
       
    def forward(self, input):
        warnings.filterwarnings('error',category=RuntimeWarning)
        try: 
            #print('Softmax forward method called')
            #Here input is the output of the forward method of the dense layer that came before
            self.input = input
            exp_input = np.exp(self.input)
            sum_exp_input = np.sum(exp_input)
            self.output = exp_input/sum_exp_input
            return self.output 
        except Warning:
            print(f"\nD'oh, overflow encountered in Softmax! Try CalcTanh instead or reduce number of hidden layers or layer size.\n")
            

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        M = np.tile(self.output, n)
        input_gradient = np.dot(M * (np.identity(n) - np.transpose(M)), output_gradient)
        self.activation_prime_output = np.zeros((n,1))
        return input_gradient


#Plotting activation functions and derivatives thereof
"""
def plot(activation_func):
    test_input = np.arange(-1,1,0.1)
    #test_input = np.arange(-1,1,0.5)
    test_input = test_input.reshape(len(test_input),1)
    #print(f'test_input is\n{test_input}\n')
    learning_rate = 0.1  
    test_output_forward = activation_func.forward(test_input)
    #print(f'test_output_forward is\n{test_output_forward}\n')
    test_output_backward = activation_func.backward(test_input,learning_rate)
    #print(f'activation_prime_output is\n{activation_func.activation_prime_output}\n')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
    ax1.set_title('Activation/Output Forward')
    ax2.set_title('Activation Backward')
    ax3.set_title('Output Backward')
    ax1.plot(test_input, test_output_forward)
    ax2.plot(test_input,activation_func.activation_prime_output)
    ax3.plot(test_input, test_output_backward)
    
    plt.show()
   
tanh = ActivationTanh()  
relu = ActivationReLu()  
lrelu = ActivationLReLu(0.05)
sigmo = ActivationSigmoid()
soft = ActivationSoftmax()
plot(lrelu)
"""
