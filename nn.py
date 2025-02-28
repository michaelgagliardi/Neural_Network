import os
from accelerate import accelerator
import torch
from data.dataset import *
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

EPOCHS = 10
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 128
LEARNING_RATE = 0.01

momentum = 0.5
log_interval = 10
random_seed = 1

train_dir = "data/fashion_mnist_images/train"
test_dir = "data/fashion_mnist_images/test"


class custom_scaling:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, sample):
        scaled_sample = (
            sample.reshape(sample.shape[0], sample.shape[1] * sample.shape[2]).astype(
                np.float32
            )
            - self.scale_factor / self.scale_factor
        )
        return scaled_sample


data_transform = transforms.Compose(
    [
        custom_scaling(scale_factor=127.5),
        transforms.Resize(size=(28, 28)),
        transforms.ToTensor(),
    ]
)

train_data = datasets.ImageFolder(
    root=train_dir, transform=data_transform, target_transform=None
)
test_data = datasets.ImageFolder(
    root=test_dir, transform=data_transform, target_transform=None
)

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=128,  # how many samples per batch?
    num_workers=4,  # how many subprocesses to use for data loading? (higher = more)
    shuffle=True,
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=128,
    num_workers=4,
    shuffle=False,
)


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # neural networks like their inputs in vector form
            nn.Linear(
                in_features=input_shape, out_features=hidden_units
            ),  # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


# See classes
class_names = train_data.classes

model_0 = FashionMNISTModelV0(
    input_shape=784,  # one for every pixel (28x28)
    hidden_units=64,  # how many units in the hidden layer
    output_shape=len(class_names),  # one for every class
)

print(train_dataloader, test_dataloader)

accuracy = self.accuracy.calculate(predictions, batch_y)

image, label = train_data[0]
print(image.shape, label)
