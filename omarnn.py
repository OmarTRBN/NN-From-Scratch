import numpy as np

'''
number of layers = 4 ---> W1, W2, W3 with size (inputs, num_of_neurons)
  //   //  //    = 4 ---> b1 , ... , bm (m is total number of neurons)

[2, 2, 3, 1] where every number is the amount of neurons
so 2 inputs, 2 hidden layers of 2, 3 and 1 output 

'''

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


def sigmoid_derivative(x):
    y = sigmoid(x)
    dy = y*(1-y)
    return dy


activations = {
    "sigmoid": sigmoid
}

activation_derivatives = {
    "sigmoid": sigmoid_derivative
}


def check_parameters_shapes(parameters):
    for key, value in parameters.items():
        print(f"{key}: {value.shape}")


class NeuralNetwork():
    def __init__(self, structure, activations, derivatives):
        self.structure = structure
        self.activations = activations  # Length must be 1 less than structure list
        self.derivatives = derivatives
        self.parameters = {}

    def initialize_parameters(self, scale=0.01):
        for i in range(len(self.structure)-1):
            self.parameters[f"W{i+1}"] = np.random.randn(
                self.structure[i], self.structure[i+1]) * scale
            self.parameters[f"b{i+1}"] = np.zeros((self.structure[i+1], 1))
        return self.parameters

    def feedforward(self, Xin):
        cache = {}
        cache["Z1"] = np.dot(self.parameters["W1"].T, Xin) + \
            self.parameters["b1"]  # Z = W'.X + b
        cache["A1"] = self.activations[0](cache["Z1"])
        for i in range(len(self.structure[1:])-1):
            cache[f"Z{i+2}"] = np.dot(self.parameters[f"W{i+2}"].T,
                                      cache[f"A{i+1}"]) + self.parameters[f"b{i+2}"]  # Z = W'A + b
            cache[f"A{i+2}"] = self.activations[i+1](cache[f"Z{i+2}"])
        return cache

    def backpropagate(self, cache, Xin, Yout, learning_rate=0.01):
        grads = {}
        deltas = {}
        n_l = len(self.structure)  # Number of layers

        deltas[f"delta{n_l-1}"] = Yout - cache[f'A{n_l-1}']  # delta = Y - A
        for i in range(n_l-2):
            deltas[f"delta{n_l-i-2}"] = np.dot(
                self.parameters[f"W{n_l-i-1}"], deltas[f"delta{n_l-i-1}"]) * self.derivatives[i](cache[f"Z{n_l-i-2}"])  # delta = (W.delta).f'(Z)

        for i in range(n_l-1):
            # dW = A.delta'
            grads[f"dW{i+1}"] = np.dot(cache[f"A{i}"], deltas[f"delta{i+1}"].T)
            # db = sum(delta)
            grads[f"db{i+1}"] = np.sum(deltas[f"delta{i+1}"],
                                       axis=1, keepdims=True)

            # Update parameters
            self.parameters[f"W{i+1}"] += learning_rate * grads[f"dW{i+1}"]
            self.parameters[f"b{i+1}"] += learning_rate * grads[f"db{i+1}"]
        return grads

    def nn_train(self, Xin, Yout, epoch=100, learning_rate=0.01):
        if len(self.parameters) == 0:
            self.parameters = self.initialize_parameters()
        for iteration in range(epoch):
            cache = self.feedforward(Xin)
            grads = self.backpropagate(
                cache, Xin, Yout, learning_rate=learning_rate)
        print("Training complete!")
        return
