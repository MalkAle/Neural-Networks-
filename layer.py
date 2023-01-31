import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.alpha = None

    def forward(self, input):
        pass

    def backward(self, error_gradientY, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.count = 0
        #Inititializing weights and biases randomly
        #print(f'Initializing Dense layer, input_size is {input_size}, output size is {output_size}')
        self.weights = np.random.randn(output_size,input_size)
        #self.biases = np.zeros((output_size,1))
        self.biases = np.random.randn(output_size,1)
        #print(f'Initial weights are\n{self.weights}\n')
        #print(f'Initial biases are\n{self.biases}\n')

    def __repr__(self):
        return "Dense"
 
    def forward(self, input):
        #print('Forward method in Dense called\n')
        self.input = input
        #Output Y = W * X + B
        #print(f'input for forward method in Dense is\n{self.input}\n')
        output = np.dot(self.weights, self.input) + self.biases
        #print(f'Output of Dense forward method is\n{output}\n')
        self.count += 1
        #print(f'forward count is {self.count}\n')
        return output

    #Output gradient is parial differential of error E in respect to output Y
    #Input gradient is parial differential of error E in respect to output X
    def backward(self, error_gradientY, learning_rate):
        #print(f'error_gradientY\n{error_gradientY}\n')
        #print(f'self.input is \n{self.input}\n')
        #print(f'self.input transposed is \n{np.transpose(self.input)}\n')
        weights_gradient = np.dot(error_gradientY,np.transpose(self.input))
        #print(f'weights_gradient is\n{weights_gradient}\n')
        #Calculating input gradient to feed in the previous layer before updating them
        error_gradientX = np.dot(np.transpose(self.weights),error_gradientY)
        #print(f'error_gradinetX is\n{error_gradientX}\n')
        #print(f'Adjusting weights and biases, learning_rate is {learning_rate}\n')
        #print(f'old weights are\n{self.weights}\n')
        #Updating weights using weight gradient and increment (learning rate) 
        self.weights -= learning_rate * weights_gradient
        #print(f'new weights are\n{self.weights}\n')
        #print(f'old biases are\n{self.biases}\n')
        #Bias gradient is equal to output gradient, so biases are uptdated as follows
        self.biases -= learning_rate * error_gradientY
        #print(f'new biases are\n{self.biases}\n')
        #print(f'Output of Dense backward method is\n{error_gradientX}\n')
        return error_gradientX