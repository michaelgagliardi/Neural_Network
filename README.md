# Beat PyTorch! Neural Network Competition App

This Streamlit application allows you to compare the performance of a custom neural network implementation against a PyTorch implementation. Both models are trained on the Fashion MNIST dataset and can classify clothing items into 10 categories.

## Implementation Alignment

The key goal of this project was to make both neural network implementations as identical as possible:

### Architecture Alignment
- **Layer Size**: Both models now use 64 neurons per hidden layer (instead of 512 in the original PyTorch model)
- **Layer Structure**: Both models use the same architecture: Input -> Dense(784,64) -> ReLU -> Dense(64,64) -> ReLU -> Dense(64,10) -> Softmax
- **Activation Functions**: Both use identical activation functions (ReLU for hidden layers, Softmax for output)

### Optimizer Alignment
- **Optimizer**: Both use Adam optimizer
- **Learning Rate**: Both use 0.005
- **Decay**: Both use 5e-5 decay
- **Beta Values**: Both use the standard Adam beta values (0.9, 0.999)
- **Epsilon**: Both use 1e-7

### Regularization Alignment
- **L2 Regularization**: Both apply L2 regularization with a lambda of 5e-4
- **Dropout**: Both can optionally use dropout with the same rate

### Device Handling
- **Device Detection**: The PyTorch model now uses the same device detection logic as in the original PyTorch model:
  ```python
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```

## Features

- Upload and classify clothing images with both models
- Compare prediction accuracy, confidence scores, and processing time
- Adjust hyperparameters to find the optimal model configuration
- Interactive visualization of confidence scores
- Train models with custom hyperparameters
- Competitive "Beat PyTorch!" presentation style

## Installation

1. Clone the repository:
```bash
git clone https://github.com/michaelgagliardi/Neural_Network
cd Neural_Network
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Required Dependencies

- streamlit
- numpy
- torch
- torchvision
- matplotlib
- opencv-python (cv2)
- pillow (PIL)

## Models

The application compares two neural network implementations:

1. **Custom Model**: A neural network built from scratch with custom implementations of:
   - Dense layers
   - Activation functions (ReLU, Softmax)
   - Optimizer (Adam)
   - Loss function (Categorical Cross Entropy)

2. **PyTorch Model**: A neural network built using the PyTorch framework with:
   - Linear layers
   - ReLU activation
   - Adam optimizer
   - CrossEntropyLoss

## Usage

1. Run the Streamlit app using the command `streamlit run app.py`
2. Upload an image of a clothing item (like a t-shirt, shoe, or dress)
3. Both models will process the image and provide their classifications
4. Compare the results to see which model performs better
5. Adjust hyperparameters in the sidebar and retrain models to improve performance

## File Structure

- `app.py`: The main Streamlit application
- `model_run.py`: Custom neural network implementation
- `pytorch_model_run.py`: PyTorch neural network implementation
- `core/`: Directory containing core components
  - `model.py`: Custom Model class
  - `pytorch_model.py`: PyTorch Model class
  - `layers.py`: Custom layers implementation
  - `activation.py`: Custom activation functions
  - `optimizer.py`: Custom optimizers
  - `loss.py`: Custom loss functions
  - `accuracy.py`: Accuracy metrics
- `models/`: Directory for saved models
  - `fashion_mnist.model`: Saved custom model
  - `pytorch_model.model`: Saved PyTorch model
- `data/`: Directory for dataset and example images

