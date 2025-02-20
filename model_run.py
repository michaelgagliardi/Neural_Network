import numpy as np
import math
from foundation import *
from optimizer import *
from dataset import create_dataset


X, y = create_dataset(points=100, classes=3)

X_test, y_test = create_dataset(points=100, classes=3)

dense1 = layer_dense(2, 64)

activation1 = activation_relu()

dense2 = layer_dense(64, 3)

loss_activation = activation_softmax_loss()

# optimizer = optimizer_sgd(learning_rate=0.85, decay=1e-3, momentum=0.9)
# optimizer = optimizer_adagrad(decay=1e-4)
# optimizer = optimizer_rmsprop(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = optimizer_adam(learning_rate=0.04, decay=5e-7)

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
              f'loss: {loss:.3f}, ' + 
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

dense1.forward(X_test)
activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis = 1)
if len(y_test.shape) ==2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print('\n')
print(f'validation acc: {accuracy:.3f}, loss: {loss:.3f}')