- For each layer the number of inputs is i and the number of outputs is j.
- This also means that the number of neurons for this layer is j. 
- General equation for a forward method in a layer is Y = W * X + B, where Y is the output and X is the input (for this particular layer).
- The datapoints of a dataset are passed one by one using a loop, so the forward and backward propagation are axercuted each time for
a single datapoint. 
- Y is a vector of the shape j x 1, where j is the number of neurons of this layer. In the output layer of the network j is the number of classes that should be predicted.
- X is a matrix of the shape i x 1 where i is the the number of inputs. In der input layer i is the number of variables in the training dataset.
- W is the weights matrix. It has the shape of j x i
- B is the bias matrix. It has the shape of j x 1