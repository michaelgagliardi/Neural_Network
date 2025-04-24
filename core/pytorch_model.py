import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import copy
import time
from core.accuracy import *

# Use the same device definition as in the original
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PyTorch_Model(nn.Module):
    def __init__(self):
        super(PyTorch_Model, self).__init__()
        self.layers = nn.ModuleList()
        self.softmax_classifier_output = None
        self.output_layer_activation = None
        self.to(device)  # Move model to the correct device

    def add(self, layer):
        self.layers.append(layer.to(device))
        # Keep track of the output layer activation (last layer)
        if isinstance(layer, nn.Softmax):
            self.output_layer_activation = layer

    def forward(self, X):
        X = X.to(device)
        for layer in self.layers:
            X = layer(X)
        return X

    def set(self, *, loss=nn.CrossEntropyLoss(), optimizer=None, accuracy=None):
        self.loss = loss.to(device)
        self.optimizer = optimizer if optimizer else optim.Adam(self.parameters())
        self.accuracy = accuracy if accuracy else accuracy_categorical(binary=False)

    def finalize(self):
        """
        Equivalent to the finalize method in custom model
        In PyTorch, we don't need to manually connect layers
        """
        pass

    def calculate_accuracy(self, input_size, data_loader):
        """Calculate accuracy on a dataset"""
        self.accuracy.new_pass()
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(-1, input_size)  # Flatten images
                outputs = self.forward(images)
                # Convert to numpy for the accuracy calculation
                outputs_np = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Use the same prediction method as custom model
                predictions = np.argmax(outputs_np, axis=1)
                self.accuracy.calculate(predictions, labels_np)

        return self.accuracy.calculate_accumulated()

    def calculate_loss(self, outputs, labels, l1_lambda=0, l2_lambda=0):
        """Calculate loss including regularization"""
        # Data loss
        data_loss = self.loss(outputs, labels)

        # Regularization loss
        regularization_loss = 0
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            regularization_loss += l1_lambda * l1_norm

        if l2_lambda > 0:
            l2_norm = sum(p.pow(2).sum() for p in self.parameters())
            regularization_loss += l2_lambda * l2_norm

        # Total loss
        total_loss = data_loss + regularization_loss

        return data_loss, regularization_loss, total_loss

    def start_train(
        self,
        input_size,
        train_dataloader,
        epochs=1,
        print_every=1,
        l1_lambda=0,
        l2_lambda=0,
        validation_data=None,
    ):
        """Training loop similar to custom model's train method"""
        self.train()  # Set model to training mode

        for epoch in range(1, epochs + 1):
            print(f"epoch: {epoch}")

            # Training loop
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)
                images = images.view(-1, input_size)  # Flatten images

                # Forward pass
                outputs = self.forward(images)

                # Calculate loss
                data_loss, regularization_loss, total_loss = self.calculate_loss(
                    outputs, labels, l1_lambda, l2_lambda
                )

                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Print progress
                if batch_idx % print_every == 0:
                    # Calculate accuracy on the current batch
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == labels).sum().item()
                    accuracy = correct / labels.size(0)

                    print(
                        f"epoch: {epoch}, "
                        + f"acc: {accuracy:.3f}, "
                        + f"loss: {total_loss.item():.4f}, ("
                        + f"data loss: {data_loss.item():.4f}, "
                        + f"regularization loss: {regularization_loss.item():.4f}), "
                        + f"lr: {self.optimizer.param_groups[0]['lr']:.5f}"
                    )

            # Evaluate on validation data if provided
            if validation_data is not None:
                self.evaluate(validation_data)

    def evaluate(self, test_dataloader):
        """Evaluation similar to custom model's evaluate method"""
        self.eval()  # Set model to evaluation mode

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (X_val, y_val) in enumerate(test_dataloader):
                X_val, y_val = X_val.to(device), y_val.to(device)
                X_val = X_val.view(X_val.size(0), -1)  # Flatten images

                # Forward pass
                outputs = self.forward(X_val)

                # Calculate loss
                loss = self.loss(outputs, y_val)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        avg_loss = total_loss / len(test_dataloader)
        accuracy = correct / total

        print(f"validation acc: {accuracy:.3f}, loss: {avg_loss:.3f}")

        # Switch back to training mode
        self.train()
        return accuracy, avg_loss

    def predict(self, X, batch_size=None):
        """Prediction similar to custom model's predict method"""
        self.eval()  # Set model to evaluation mode

        # Make sure X is on the correct device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(device)
        else:
            X = X.to(device)

        # Handle batch prediction
        if batch_size is None:
            # Single batch prediction
            with torch.no_grad():
                outputs = self.forward(X)
            return outputs
        else:
            # Multiple batch prediction
            outputs = []
            n_samples = X.size(0)
            n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

            with torch.no_grad():
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n_samples)
                    batch_X = X[start_idx:end_idx]
                    batch_output = self.forward(batch_X)
                    outputs.append(batch_output)

            return torch.cat(outputs, dim=0)

    def save(self, path):
        """Save model to file"""
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict from file"""
        super().load_state_dict(state_dict, strict)
        # Set output_layer_activation after loading
        for layer in reversed(self.layers):
            if isinstance(layer, nn.Softmax):
                self.output_layer_activation = layer
                break
        return self

    @staticmethod
    def load(path, *args, **kwargs):
        """Load model from file"""
        model = PyTorch_Model(*args, **kwargs)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model
