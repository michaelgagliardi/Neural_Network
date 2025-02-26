import numpy as np
import math
import sys
import matplotlib

class layer_dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_L1=0, weight_regularizer_L2=0,
                 bias_regularizer_L1=0, bias_regularizer_L2=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        ##set regularization strength
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        ##gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        if self.weight_regularizer_L2 > 0:
            dL2 = np.ones_like(self.weights)
            dL2[self.weights < 0] = -1
            self.dweights += 2 * self.weight_regularizer_L2 * dL2

        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1
        if self.bias_regularizer_L2 > 0:
            dL2 = np.ones_like(self.biases)
            dL2[self.biases < 0] = -1
            self.dbiases += 2 * self.bias_regularizer_L2 * dL2

        ##gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class dropout_layer:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        self.binary_mask = (
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        )
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class layer_input:
    def forward(self, inputs, training):
        self.output = inputs
