import numpy as np
import math
import sys
import matplotlib
from core.layers import *
from core.activation import *

class Loss:

    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
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

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):

        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

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
        samples = len(dvalues)
        labels = len(dvalues[0])
        # handle sparse labels, turn in one-hot encoded
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues

        # normalize
        self.dinputs = self.dinputs / samples


##softmax classifier -- combined softmax activation and cross-entropy loss
class activation_softmax_loss(Loss):
    ## below functions not needed for current implementation
    # def __init__(self):
    #     self.activation = activation_softmax()
    #     self.loss = loss_categorical_cross_entropy()

    # def forward(self, inputs, y_true):
    #     ##output layer's activation function
    #     self.activation.forward(inputs)
    #     self.output = self.activation.output
    #     return self.loss.calculate(self.output, y_true)

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


class mse_loss(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class mae_loss(Loss):
    def forward(self, y_pred, y_true):
        samples_losses = np.mean(np.abs(y_true - y_pred), axis = -1)

        return samples_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
