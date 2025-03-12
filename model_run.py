import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from data.dataset import *
from core.layers import *
from core.activation import *
from core.accuracy import *
from core.optimizer import *
from core.loss import *
from core.model import Model

EPOCHS = 10
BATCH_SIZE = 128
LAYER_SIZE = 64
LEARNING_RATE = 0.005

fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

model_type = "mnist"

if model_type == "regression":
    X, y = sine_dataset()

    model = Model()
    model.add(layer_dense(1, LAYER_SIZE))
    model.add(activation_relu())
    model.add(layer_dense(LAYER_SIZE, LAYER_SIZE))
    model.add(activation_relu())
    model.add(layer_dense(LAYER_SIZE, 1))
    model.add(activation_linear())

    model.set(
        loss=mse_loss(),
        optimizer=optimizer_adam(learning_rate=LEARNING_RATE, decay=1e-3),
        accuracy=accuracy_regression(),
    )

    model.finalize()

    model.train(X, y, epochs=10000, print_every=100)


elif model_type == "categorical":
    X, y = spiral_dataset(1000, 3)
    X_test, y_test = spiral_dataset(100, 3)

    model = Model()

    model.add(layer_dense(2, 512, weight_regularizer_L2=5e-4, bias_regularizer_L2=5e-4))
    model.add(activation_relu())
    model.add(dropout_layer(0.1))
    model.add(layer_dense(512, 3))
    model.add(activation_softmax())
    model.set(
        loss=loss_categorical_cross_entropy(),
        optimizer=optimizer_adam(learning_rate=LEARNING_RATE, decay=5e-5),
        accuracy=accuracy_categorical(),
    )

    model.finalize()

    model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)

elif model_type == "mnist":
    X, y, X_test, y_test = create_data_mnist("data/fashion_mnist_images")

    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)

    X = X[keys]
    y = y[keys]

    X = (
        X.reshape(X.shape[0], X.shape[1] * X.shape[2]).astype(np.float32) - 127.5
    ) / 127.5

    X_test = (
        X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).astype(
            np.float32
        )
        - 127.5
    ) / 127.5

    model = Model()

    model.add(layer_dense(X.shape[1], LAYER_SIZE))
    model.add(activation_relu())
    model.add(layer_dense(LAYER_SIZE, LAYER_SIZE))
    model.add(activation_relu())
    model.add(layer_dense(LAYER_SIZE, 10))
    model.add(activation_softmax())

    model.set(
        loss=loss_categorical_cross_entropy(),
        optimizer=optimizer_adam(learning_rate=LEARNING_RATE, decay=5e-5),
        accuracy=accuracy_categorical(),
    )

    model.finalize()

    model.train(
        X,
        y,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        print_every=100,
    )

    model.save_parameters("parameters/fashion_mnist.parms")
    # New model
    # Instantiate the model
    model = Model()
    # Add layers
    model.add(layer_dense(X.shape[1], LAYER_SIZE))
    model.add(activation_relu())
    model.add(layer_dense(LAYER_SIZE, LAYER_SIZE))
    model.add(activation_relu())
    model.add(layer_dense(LAYER_SIZE, 10))
    model.add(activation_softmax())
    model.set(loss=loss_categorical_cross_entropy(), accuracy=accuracy_categorical())
    # Finalize the model
    model.finalize()
    # Set model with parameters instead of training it
    model.load_parameters("parameters/fashion_mnist.parms")
    # Evaluate the model
    model.evaluate(X_test, y_test)

    model.save("models/fashion_mnist.model")

elif model_type == "load":
    # Load the model
    image_data = cv2.imread("data/tshirt.png", cv2.IMREAD_GRAYSCALE)
    image_data = cv2.resize(image_data, (28, 28))
    image_data = 255 - image_data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
    model = Model.load("models/fashion_mnist.model")
    # Predict on the image
    confidences = model.predict(image_data)
    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)
    # Get label name from label index
    prediction = fashion_mnist_labels[predictions[0]]
    print(prediction)
else:
    print('Please specify model type as "regression" or "categorical"')
