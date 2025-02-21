import numpy as np
import math
import sys
import matplotlib
from foundation import *
from activation import *

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer: layer_dense):
        regularization_loss = 0
        # weight regularization
        if layer.weight_regularizer_L1 > 0:
            regularization_loss += layer.weight_regularizer_L1 * (
                np.sum(np.abs(layer.weights))
            )
        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * (
                np.sum(layer.weights * layer.weights)
            )
        # bias regularization
        if layer.bias_regularizer_L1 > 0:
            regularization_loss += layer.bias_regularizer_L1 * (
                np.sum(np.abs(layer.biases))
            )
        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * (
                np.sum(layer.biases * layer.biases)
            )
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
        # handle sparse labels, turn in one-hot encoded
        if len(y_true.shape) == 1:
            y_true = np.eye(len(dvalues[0]))[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues

        # normalize
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
        # turn one-hot encoded labels into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        ##calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        ##normalize
        self.dinputs = self.dinputs / samples


class binary_crossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = (
            -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        )

        self.dinputs = self.dinputs / samples
