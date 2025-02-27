import numpy as np
import math


class activation_relu:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        ##make zero gradient where inputs are < 0
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class activation_softmax:

    def forward(self, inputs, training):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            ##flatten
            output = output.reshape(-1, 1)
            ##calculate jacobian matrix
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            ##samplewise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, dvalue)
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis = 1)


class activation_sigmoid:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class activation_linear:
    def predictions(self, outputs):
        return outputs
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
