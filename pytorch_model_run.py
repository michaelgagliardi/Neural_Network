from data.dataset import *
from core.accuracy import *
from core.pytorch_model import *
from core.accuracy import *

from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import time


image_path_list = list(Path("data/fashion_mnist_images").glob("*/*/*.png"))
train_dir = "data/fashion_mnist_images/train"
test_dir = "data/fashion_mnist_images/test"

# Set device for PyTorch - match the device from pytorch_model.py
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


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

# Match the hyperparameters exactly from the custom model
EPOCHS = 10
NUM_WORKERS = 4
PRINT_EVERY = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.005
LAYER_SIZE = 64  # Changed from 512 to 64 to match custom model
DECAY = 5e-5  # Match decay rate from custom model
L2_LAMBDA = 5e-4  # Match L2 regularization from custom model (categorical example)
DROPOUT_RATE = 0.0  # No dropout in the main model

if __name__ == "__main__":
    data_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                size=(28, 28)
            ),  # Resize to 28x28 (matching custom nn data inputs)
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize between -1 and 1 (matching custom nn data inputs)
        ]
    )

    # Load the datasets
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

    print(
        f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers."
    )

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

    # Add hidden layers and ReLU activations - matching custom model architecture
    model.add(
        nn.Linear(in_features=input_size, out_features=LAYER_SIZE)
    )  # 28x28 input images
    model.add(nn.ReLU())
    model.add(nn.Linear(in_features=LAYER_SIZE, out_features=LAYER_SIZE))
    model.add(nn.ReLU())
    model.add(nn.Linear(in_features=LAYER_SIZE, out_features=output_size))
    model.add(nn.Softmax(dim=1))

    # Set loss and optimizer - exactly matching custom model settings
    model.set(
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=DECAY,
            betas=(0.9, 0.999),
            eps=1e-7,
        ),
    )

    # Train the model
    start_time = time.time()

    model.start_train(
        input_size=input_size,
        train_dataloader=train_dataloader,
        print_every=PRINT_EVERY,
        epochs=EPOCHS,
        validation_data=test_dataloader,
        l1_lambda=0,  # No L1 regularization in the main model
        l2_lambda=L2_LAMBDA,  # Match L2 regularization from custom model
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate the model
    model.evaluate(test_dataloader)

    # Save the model
    model.save("./models/pytorch_model.model")

    # Load the model for prediction
    model = PyTorch_Model()
    model.add(nn.Linear(in_features=input_size, out_features=LAYER_SIZE))
    model.add(nn.ReLU())
    model.add(nn.Linear(in_features=LAYER_SIZE, out_features=LAYER_SIZE))
    model.add(nn.ReLU())
    model.add(nn.Linear(in_features=LAYER_SIZE, out_features=output_size))
    model.add(nn.Softmax(dim=1))
    model.load_state_dict(
        torch.load("./models/pytorch_model.model", map_location=device)
    )
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Example prediction
    test_image_path = "./data/tshirt.png"  # Update with the correct path
    if Path(test_image_path).exists():
        # Load and preprocess the image
        test_image = load_image(test_image_path, data_transform)
        # Flatten the image
        test_image = test_image.view(test_image.size(0), -1)

        # Predict the class
        with torch.no_grad():  # Disable gradient calculation
            prediction = model.predict(test_image)
        predicted_class = prediction.argmax(dim=1).item()
        print(f"Predicted class: {fashion_mnist_labels[predicted_class]}")
    else:
        print(f"Test image not found at {test_image_path}")
