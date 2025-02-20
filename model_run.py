import numpy as np
import math
from layers import *
from optimizer import optimizer_sgd


X, y = create_dataset(points=100, classes=3)

dense1 = layer_dense(2, 64)

activation1 = activation_relu()

dense2 = layer_dense(64, 3)

loss_activation = activation_softmax_loss()

optimizer = optimizer_sgd()

for epoch in range(10001):
    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + 
              f'acc: {accuracy:.3f}, ' + 
              f'loss: {loss:.3f}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

