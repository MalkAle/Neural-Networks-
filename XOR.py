import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)


if __name__ == '__main__':
    class Dense:
        def __init__(self, input_size, output_size):
            #Inititializing weights and biases randomly
            self.weights = np.random.randn(output_size,input_size)
            self.biases = np.random.randn(output_size,1)
            print(f'Initial weights are\n{self.weights}\n')
            print(f'Initial biases are\n{self.biases}\n')
        #Forward method calculates Y, later this will be compared with the desired output Y* to calculate error E 
        def forward(self, input):
            self.input = input
            #Output Y = W * X + B
            output = np.dot(self.weights, self.input) + self.biases
            return output

        #Output gradient is parial differential of error E in respect to output Y
        #Input gradient is parial differential of error E in respect to output X
        def backward(self, error_gradientY, learning_rate):
            weights_gradient = np.dot(error_gradientY,self.input.T)
            #Calculating input gradient to feed in the previous layer before updating them
            error_gradientX = np.dot(self.weights.T,error_gradientY)
            #Updating weights using weight gradient and increment (learning rate) 
            self.weights -= learning_rate * weights_gradient
            #Bias gradient is equal to output gradient, so biases are uptdated as follows
            self.biases -= learning_rate * error_gradientY
            return error_gradientX

    class Activation:
        #"activation" and "activation_prime" are functions
        def __init__(self, activation, activation_prime):
            self.activation = activation
            self.activation_prime = activation_prime

        #Forward method simply applies the activation function to the input f(X)
        def forward(self, input):
            self.input = input
            return self.activation(self.input)

        #Backward method calculates the input gradient to feed in the previous layer
        #through Hadamard Product of output_gradient and activaton_prime
        def backward(self, output_gradient,learning_rate):
            input_gradient = np.multiply(output_gradient, self.activation_prime(self.input))
            return input_gradient

    class ActivationTanh(Activation):
        def __init__(self):
            #lambda fuction is used to pass functions like arguments
            tanh = lambda x: np.tanh(x)
            tanh_prime = lambda x: 1 - np.tanh(x) ** 2
            #Initializing the parent class with the attributes tanh and tanh_prime as functions
            #Calling super method to initialize the parent class Activation to use it's forward and backward functions
            super().__init__(tanh, tanh_prime)

    
    def mse(y_true, y):
        return np.mean((y_true - y)**2)

    def mse_prime(y_true, y):
        return 2 * (y - y_true) / np.size(y_true)

    #In this training dataset i = 2 and n = 4 (2 variables and 4 observations)
    #Therefore shape of X is 2 x 4
    #Also, each datapoint for X needs to be a vector of the shape 2 x 1
    #Then the datapoints are passed in a vector of the shape n x (2 x 1), where n is the number of datapoints
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    #And shape of Y is 1 x 4
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    #Since we have 2 variables first layer has 2 neurons.
    #Second layer could have any number of layers, hare we define 3.
    #Output layer has 1 neuron since output of the whole network is either 0 or 1.

    d1 = Dense(2, 3)
    #For this layer shapes are as follows
    #Y: 3 x 1
    #W: 3 x 2
    #X: 2 x 4
    #B: 3 x 1
    a1 = ActivationTanh()

    #Output layer has input_size of 3 (from 3 neurons of previous layer, i = 3) 
    #and output_size of 1 since it has 1 neuron, (j = 1)
    d2 = Dense(3, 1)
    #For this layer shapes are as follows
    #Y: 1 x 1
    #W: 1 x 3
    #X: 3 x 4
    #B: 1 x 1
    a2 = ActivationTanh()

    network = [d1, a1, d2, a2]

    epochs = 1000
    learning_rate = 0.1
    error_tracking = []

    for epoch in range(epochs):
        error = 0 
        #print('new iteration------------------------------------------------------')
        for x,y in zip(X,Y):
            #Set 0 for each iteration
            layer_output = x
            for layer in network:
                layer_output = layer.forward(layer_output)

            network_output = layer_output #to make it extra clear
            error += mse(y, network_output) # sums up mse for each iteration
            #print(f'error is {mse(y, network_output)}\n')
            #print(f'error sum is {error}\n')
            error_gradientY = mse_prime(y, network_output)

            layer_input = error_gradientY
            for layer in reversed(network):
                layer_input = layer.backward(layer_input, learning_rate)
            
        error /=len(X)
        error_tracking.append(error)
        print(f'Mean error for this iteration is {error}')

    print(f'Final mean error is {error}')

    print(f'\nfinal d1 weights are\n{d1.weights}\n')
    print(f'final d1 biases are\n{d1.biases}\n')
    print(f'final d2 weights are\n{d2.weights}\n')
    print(f'final d2 biases are\n{d2.biases}\n')

    test_data = np.reshape([[0.5, 0.2]], (2, 1))

    layer_output = test_data
    for layer in network:
        layer_output = layer.forward(layer_output)

    plt.plot(np.arange(0,epochs,1), error_tracking)
    plt.xlabel('Epoch')
    #plt.title('Title')
    plt.ylabel('Error (mse, mean per iteration)')
    plt.show()

    print(f'Prediction for the input is \n{layer_output}\n')


