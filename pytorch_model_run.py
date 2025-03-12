import os
from accelerate import accelerator
import random
from PIL import Image
from data.dataset import *
from pathlib import Path
from core.accuracy import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_model import *
from core.accuracy import *

image_path_list = list(Path("data/fashion_mnist_images").glob("*/*/*.png"))
train_dir = "data/fashion_mnist_images/train"
test_dir = "data/fashion_mnist_images/test"

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
NUM_WORKERS = 0

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
LAYER_SIZE = 64
L1_LAMBDA = 5e-5

data_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            size=(28, 28)
        ),  ##resize to 28x28 (matching custom nn data inputs)
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,), (0.5,)
        ),  ##normalize between -1 and 1 (matching custom nn data inputs)
    ]
)

train_data = datasets.ImageFolder(
    root=train_dir, transform=data_transform, target_transform=None
)
test_data = datasets.ImageFolder(
    root=test_dir, transform=data_transform, target_transform=None
)

sample_image, _ = train_data[0]  # Get the first sample (image, label)
input_size = sample_image.numel()  # Flattened size of the image

# Get the output size dynamically by checking the number of classes
output_size = len(train_data.classes)

print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
)

test_dataloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
)


# Initialize the model
model = PyTorch_Model()

# Add hidden layers and ReLU activations
model.add(nn.Linear(in_features=input_size, out_features=LAYER_SIZE))  # 28x28 input images
model.add(nn.ReLU())
model.add(nn.Linear(in_features=LAYER_SIZE, out_features=LAYER_SIZE))
model.add(nn.ReLU())
model.add(nn.Linear(in_features=LAYER_SIZE, out_features=output_size))
model.add(nn.Softmax(dim=1))

# Output layer
model.set(loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE))

model.train(input_size=input_size, train_dataloader=train_dataloader)

model.evaluate(test_dataloader)
