import numpy as np

### Activation functions ###
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def sigmoid_derivative(x):
    y = sigmoid(x)
    dy = y*(1-y)
    return dy
############################

###   Loss Functions  ###
def loss(e, y): 
    # e is expected, y is model's output
    loss = np.sum((y-e)**2)
    return loss
############################

class Neuron():
    def __init__(self, num_of_weights):
        self.weights = np.random.randn(num_of_weights, 1) * 0.01
        self.bias = np.zeros((1,1))

class NeuralNetwork():
    def __init__(self, number_of_hidden_layers):
        self.layers = [] # This currently has the name of every girl I will ever date
        self.number_of_hidden_layers = number_of_hidden_layers 
        self.activations = [] # This will have every activation in every layer
    
    def construct(self):
        n_x = input("Enter number of inputs: ")
        self.layers.append(n_x)
        for layer in range(self.number_of_hidden_layers):
            neuron_list = []
            num_of_neurons = input(f"Enter number of neurons for layer {layer+1}: ")
            for i in range(num_of_neurons):
                neuron_list.append(Neuron())
            self.hidden_layers.append(neuron_list)
    
    def predict(inputs, weights, biases):
        pass