import numpy as np
import math
from foundation import *
from optimizer import *
from dataset import spiral_dataset
from activation import *
from loss import *

##create dataset
X, y = spiral_dataset(points=1000, classes=2)

y=y.reshape(-1,1)

##define model and hyperparameters
dense1 = layer_dense(2, 64, weight_regularizer_L2 = 5e-4, bias_regularizer_L2=5e-4)

activation1 = activation_relu()

dense2 = layer_dense(64, 1)

activation2 = activation_sigmoid()

# dropout1 = dropout_layer(0.1)

# loss_activation = activation_softmax_loss()

loss_function = binary_crossentropy()

# optimizer = optimizer_sgd(learning_rate=0.85, decay=1e-3, momentum=0.9)
# optimizer = optimizer_adagrad(decay=1e-4)
# optimizer = optimizer_rmsprop(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = optimizer_adam(decay=5e-7)

##train model
for epoch in range(10001):
    dense1.forward(X)

    activation1.forward(dense1.output)

    # dropout1.forward(activation1.output)

    dense2.forward(activation1.output)

    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output, y)

    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    # predictions = np.argmax(loss_function.output, axis=1)
    # if len(y.shape) == 2:
    #     y = np.argmax(y, axis=1)
    # accuracy = np.mean(predictions == y)

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, "
            + f"acc: {accuracy:.3f}, "
            + f"loss: {loss:.3f}, ("
            + f"regularization loss: {regularization_loss:.3f}, "
            + f"data loss: {data_loss:.3f}), "
            + f"lr: {optimizer.current_learning_rate:.5f}"
        )

    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    # dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

##test the model
X_test, y_test = spiral_dataset(points=100, classes=2)

y_test = y_test.reshape(-1, 1)
dense1.forward(X_test)
activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y_test)

# predictions = np.argmax(loss_function.output, axis = 1)
# if len(y_test.shape) ==2:
#     y_test = np.argmax(y_test, axis=1)
# accuracy = np.mean(predictions == y_test)
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print('\n')
print(f'validation acc: {accuracy:.3f}, loss: {loss:.3f}')
