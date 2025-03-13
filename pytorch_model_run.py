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


image_path_list = list(Path("data/fashion_mnist_images").glob("*/*/*.png"))
train_dir = "data/fashion_mnist_images/train"
test_dir = "data/fashion_mnist_images/test"


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

EPOCHS = 10
NUM_WORKERS = 4
PRINT_EVERY = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.005
LAYER_SIZE = 512
L1_LAMBDA = 5e-4

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

    # # Initialize the model
    # model = PyTorch_Model()

    # # Add hidden layers and ReLU activations
    # model.add(
    #     nn.Linear(in_features=input_size, out_features=LAYER_SIZE)
    # )  # 28x28 input images
    # model.add(nn.ReLU())
    # model.add(nn.Linear(in_features=LAYER_SIZE, out_features=LAYER_SIZE))
    # model.add(nn.ReLU())
    # model.add(nn.Linear(in_features=LAYER_SIZE, out_features=output_size))
    # model.add(nn.Softmax(dim=1))

    # # Set loss and optimizer
    # model.set(
    #     loss=nn.CrossEntropyLoss(),
    #     optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE),
    # )

    # # Train the model
    # model.start_train(
    #     input_size=input_size,
    #     train_dataloader=train_dataloader,
    #     print_every=PRINT_EVERY,
    #     epochs=EPOCHS
    # )

    # # Evaluate the model
    # model.evaluate(test_dataloader)

    # # Save the model
    # model.save("./models/pytorch_model.model")

    # Load the model
    model = PyTorch_Model()
    model.add(
        nn.Linear(in_features=input_size, out_features=LAYER_SIZE)
    )  # 28x28 input images
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

    # Load and preprocess the image of a T-shirt
    tshirt_image_path = "./data/fashion_mnist_images/test/4/0000.png"  # Update with the correct path
    tshirt_image = load_image(tshirt_image_path, data_transform)
    # Flatten the image
    tshirt_image = tshirt_image.view(tshirt_image.size(0), -1)

    # Predict the class of the T-shirt image
    with torch.no_grad():  # Disable gradient calculation
        prediction = model.predict(tshirt_image)
    predicted_class = prediction.argmax(dim=1).item()
    print(f"Predicted class: {fashion_mnist_labels[predicted_class]}, {prediction}")
