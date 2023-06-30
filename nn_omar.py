import numpy as np

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def sigmoid_derivative(x):
    y = sigmoid(x)
    dy = y*(1-y)
    return dy
'''

number of layers = 4 ---> W1, W2, W3 with size (inputs, num_of_neurons)
  //   //  //    = 4 ---> b1 , ... , bm (m is total number of neurons)

[2, 2, 3, 1] where every number is the amount of neurons
so 2 inputs, 2 hidden layers of 2, 3 and 1 output 

'''

def initialize_parameters(nn_structure):
    if type(nn_structure) != list:
        nn_structure = list(nn_structure)
    parameters = {}
    for i in range(len(nn_structure)-1):
        parameters[f"W{i+1}"] = np.random.randn(
            nn_structure[i], nn_structure[i+1]) * 0.01
        parameters[f"b{i+1}"] = np.zeros((nn_structure[i+1], 1))
    return parameters

structure = [3,5,6,2,4,3,1]
parameters = initialize_parameters(structure)
def check_parameters_shapes(parameters):
    for key, value in parameters.items():
        print(f"{key}: {value.shape}")


class NeuralNetwork():
    def __init__(self, structure, activations):
        self.structure = structure
        self.activations = activations # Length must 1 less than structure list
        self.parameters = {}

    def initialize_parameters(self, scale=0.01):
        for i in range(len(self.structure)-1):
            self.parameters[f"W{i+1}"] = np.random.randn(self.structure[i], self.structure[i+1]) * scale
            self.parameters[f"b{i+1}"] = np.zeros((self.structure[i+1], 1))
        return self.parameters

    def feedforward(self, parameters, Xin):
        cache = {}
        cache["Z1"] = np.dot(parameters["W1"].T, Xin) + \
            parameters["b1"] # Z = W'.X + b 
        cache["A1"] = self.activations[0](cache["Z1"]) 
        for i in range(len(self.structure[1:])-1):
            cache[f"Z{i+2}"] = np.dot(parameters[f"W{i+2}"].T,
                                      cache[f"A{i+1}"]) + parameters[f"b{i+2}"]  # Z = W'A + b
            cache[f"A{i+2}"] = self.activations[i+1](cache["Z2"])
        return cache

    def backpropogate(self):
        pass

    def nn_train(self, parameters, epoch=100, learning_rate=0.01):
        pass

# check_parameters_shapes(parameters)
print(parameters)
