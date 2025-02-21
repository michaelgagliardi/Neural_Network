import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

np.random.seed(0)

def spiral_dataset(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

def sine_dataset(points=1000):

    X = np.arange(points).reshape(-1, 1) / points
    y = np.sin(2 * np.pi * X).reshape(-1, 1)

    return X, y


def vertical_dataset(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes)
    for class_num in range(classes):
        i = range(points * class_num, points * (class_num + 1))
        X[i] = np.c_[
            np.random.randn(points) * 0.1 + (class_num) / 3,
            np.random.randn(points) * 0.1 + 0.5,
        ]
        y[i] = class_num
    return X, y
