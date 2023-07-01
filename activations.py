import numpy as np

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