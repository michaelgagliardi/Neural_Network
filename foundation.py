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
    def forward(self, inputs):
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

class activation_relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        ##make zero gradient where inputs are < 0
        self.dinputs[self.inputs <= 0] = 0

class activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            ##flatten
            output = output.reshape(-1,1)
            ##calculate jacobian matrix
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            ##samplewise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, dvalue)
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    def regularization_loss(self, layer:layer_dense):
        regularization_loss = 0
        #weight regularization
        if layer.weight_regularizer_L1 > 0:
            regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)
        #bias regularization
        if layer.bias_regularizer_L1 > 0:
            regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases * layer.biases)
        return regularization_loss


class loss_categorical_cross_entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        #handle sparse labels, turn in one-hot encoded
        if len(y_true.shape) == 1:
            y_true = np.eye(len(dvalues[0]))[y_true]
        
        #calculate gradient
        self.dinputs = -y_true / dvalues
        
        #normalize
        self.dinputs = self.dinputs / (len(dvalues))

##softmax classifier -- combined softmax activation and cross-entropy loss
class activation_softmax_loss(Loss):
    def __init__(self):
        self.activation = activation_softmax()
        self.loss = loss_categorical_cross_entropy()
    
    def forward(self, inputs, y_true):
        ##output layer's activation function
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #turn one-hot encoded labels into discrete values
        if len(y_true.shape) ==2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        ##calculate gradient
        self.dinputs[range(samples), y_true] -=1
        ##normalize
        self.dinputs = self.dinputs / samples

