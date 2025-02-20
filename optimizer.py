import numpy as np
import math
import sys
from layers import layer_dense

class optimizer_sgd:
    def __init__(self, learning_rate = 1):
        self.learning_rate = learning_rate

    def update_params(self, layer: layer_dense):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases