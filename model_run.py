import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from dataset import *
from layers import *
from optimizer import *
from activation import *
from loss import *
from accuracy import *
from model import *

EPOCHS = 10
BATCH_SIZE = 128

fashion_mnist_labels = {
      0: 'T-shirt/top',
      1: 'Trouser',
      2: 'Pullover',
      3: 'Dress',
      4: 'Coat',
      5: 'Sandal',
      6: 'Shirt',
      7: 'Sneaker',
      8: 'Bag',
      9: 'Ankle boot'
  }

model_type = ""

if model_type == 'regression':
    X, y = sine_dataset()

    model = Model()
    model.add(layer_dense(1, 64))
    model.add(activation_relu())
    model.add(layer_dense(64,64))
    model.add(activation_relu())
    model.add(layer_dense(64,1))
    model.add(activation_linear())

    model.set(loss= mse_loss(), optimizer=optimizer_adam(learning_rate=0.005, decay=1e-3), accuracy=accuracy_regression())

    model.finalize()

    model.train(X, y, epochs=10000, print_every=100)


elif model_type == 'categorical':
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
        optimizer=optimizer_adam(learning_rate=0.05, decay=5e-5),
        accuracy=accuracy_categorical(),
    )

    model.finalize()

    model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)

elif model_type == "mnist":
    # X,y,X_test,y_test = create_data_mnist('fashion_mnist_images')
    # keys = np.array(range(X.shape[0]))
    # np.random.shuffle(keys)

    # X = X[keys]
    # y = y[keys]

    # X = (X.reshape(X.shape[0], X.shape[1]*X.shape[2]).astype(np.float32) - 127.5) / 127.5

    # X_test = (X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).astype(np.float32) - 127.5) / 127.5

    # model = Model()

    # model.add(layer_dense(X.shape[1],64))
    # model.add(activation_relu())
    # model.add(layer_dense(64,64))
    # model.add(activation_relu())
    # model.add(layer_dense(64, 10))
    # model.add(activation_softmax())

    # model.set(loss=loss_categorical_cross_entropy(),
    #           optimizer=optimizer_adam(decay=5e-5),
    #           accuracy=accuracy_categorical())

    # model.finalize()

    # model.train(X, y, validation_data=(X_test, y_test), epochs=5, batch_size=128, print_every=100)

    # model.save_parameters("fashion_mnist.parms")
    # # New model
    # # Instantiate the model
    # model = Model()
    # # Add layers
    # model.add(layer_dense(X.shape[1], 128))
    # model.add(activation_relu())
    # model.add(layer_dense(128, 128))
    # model.add(activation_relu())
    # model.add(layer_dense(128, 10))
    # model.add(activation_softmax())
    # model.set(
    # loss=loss_categorical_cross_entropy(),
    # accuracy=accuracy_categorical() )
    # # Finalize the model
    # model.finalize()
    # # Set model with parameters instead of training it
    # model.load_parameters("fashion_mnist.parms")
    # # Evaluate the model
    # model.evaluate(X_test, y_test)

    # model.save('fashion_mnist.model')
    X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
    # Scale and reshape samples
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
                127.5) / 127.5
    # Load the model
    model = Model.load('fashion_mnist.model')
    # Predict on the first 5 samples from validation dataset
    # and print the result
    confidences = model.predict(X_test[:5])
    predictions = model.output_layer_activation.predictions(confidences)
    for prediction in predictions:
        print(fashion_mnist_labels[prediction])
    print(y_test[:5])

else:

    image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)
    image_data = cv2.resize(image_data, (28, 28))
    image_data = 255 - image_data
    image_data = (image_data.reshape(1,-1).astype(np.float32)- 127.5) / 127.5
    model = Model.load('fashion_mnist.model')
    # Predict on the image
    confidences = model.predict(image_data)
    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)
    # Get label name from label index
    prediction = fashion_mnist_labels[predictions[0]]
    print(prediction)
    # print('Please specify model type as "regression" or "categorical"')
