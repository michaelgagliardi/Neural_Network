import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import copy
from core.accuracy import *

class PyTorch_Model(nn.Module):
    def __init__(self):
        super(PyTorch_Model, self).__init__()
        self.layers = nn.ModuleList()
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def set(self, *, loss=nn.CrossEntropyLoss(), optimizer=None, accuracy=None):
        self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.parameters())
        if accuracy is not None:
            self.accuracy = accuracy
        else:
            self.accuracy = accuracy_categorical(binary=False)

    def calculate_accuracy(self, input_size, data_loader, accuracy_metric):
        accuracy_metric.new_pass()
        with torch.no_grad():
            for images, labels in data_loader:
                # images, labels = images.to(device), labels.to(device)
                images = images.view(-1, input_size)  # Flatten images
                outputs = self.forward(images)
                _, predicted = torch.max(outputs, 1)  # Get class with highest score
                accuracy_metric.calculate(
                    predicted.cpu().numpy(), labels.cpu().numpy()
                )  # Use the custom accuracy class
        return accuracy_metric.calculate_accumulated()

    def train(self, input_size, train_dataloader, epochs=1, print_every=1, l1_lambda=0.0001):
        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')

            for batch_idx, (images, labels) in enumerate(train_dataloader):
            
                images = images.view(-1, input_size)  # Flatten images to 1D

                # Forward pass
                outputs = self.forward(images)
                data_loss = self.loss(outputs, labels)

                ##calculate regularization loss (not standard in pytorch)
                l2_norm = sum(p.pow(2).sum() for p in self.parameters())
                l1_norm = sum(p.abs().sum() for p in self.parameters())  # Optional for L1 regularization
                regularization_loss = l1_lambda * l1_norm + 0.01 * l2_norm

                # Total loss (data loss + regularization loss)
                total_loss = data_loss + regularization_loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if not epoch % print_every:
                    accuracy = self.calculate_accuracy(
                        input_size, train_dataloader, self.accuracy
                    )
                    print(
                        f"batch: {batch_idx}, "
                        + f"acc: {accuracy:.3f}, "
                        + f"loss: {total_loss}, ("
                        + f"data loss: {data_loss}, "
                        + f"regularization loss: {regularization_loss}), "
                        + f"lr: {self.optimizer.param_groups[0]['lr']:.5f}"
                    )

    def evaluate(self, test_dataloader):
        self.eval()
        for batch_idx, (X_val, y_val) in enumerate(test_dataloader):
            with torch.no_grad():
                output = self.forward(X_val)
                validation_loss = self.loss(output, y_val)
                predictions = output.argmax(dim=1)
                correct_preds = (predictions == y_val).sum().item()
                validation_accuracy = correct_preds / y_val.size(0)
                print(
                    f"validation acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}"
                )
        self.train()

    def predict(self, X, batch_size=None):
        self.eval()
        outputs = []
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size : (step + 1) * batch_size]

            output = self.forward(batch_X)
            outputs.append(output)

        return torch.cat(outputs)

    def save(self, path):
        model = copy.deepcopy(self)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            saved_model = pickle.load(f)
        return saved_model
