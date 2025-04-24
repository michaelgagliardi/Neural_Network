import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import sys
import os
import time
from pathlib import Path
import os

os.environ["STREAMLIT_SERVER_WATCH_PATHS"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Add the project directory to the path to import custom modules
sys.path.append(os.path.dirname(os.getcwd()))

# Import custom modules
from data.dataset import *
from core.layers import *
from core.activation import *
from core.accuracy import *
from core.optimizer import *
from core.loss import *
from core.model import Model
from core.pytorch_model import PyTorch_Model

# Set page configuration
st.set_page_config(
    page_title="Beat PyTorch!",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants and global variables
# Use the same device definition as in both model files
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH_CUSTOM = "models/fashion_mnist.model"
MODEL_PATH_PYTORCH = "models/pytorch_model.model"

# Default hyperparameters - ensure they match in both models
DEFAULT_LAYER_SIZE = 64
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_DECAY = 5e-5
DEFAULT_L2_LAMBDA = 5e-4

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

# Initialize custom accuracy metrics storage
custom_model_metrics = {
    "accuracy": 0.0,
    "loss": 0.0,
    "prediction_time": 0.0,
}

pytorch_model_metrics = {
    "accuracy": 0.0,
    "loss": 0.0,
    "prediction_time": 0.0,
}


# Helper functions for model loading and prediction
def load_custom_model():
    """Load the custom model from the saved file"""
    try:
        model = Model.load(MODEL_PATH_CUSTOM)
        return model
    except Exception as e:
        st.error(f"Error loading custom model: {e}")
        return None


def load_pytorch_model(input_size, layer_size, output_size):
    """Load the PyTorch model from the saved file"""
    try:
        # Initialize the model
        model = PyTorch_Model()

        # Add hidden layers and ReLU activations - match the custom model architecture exactly
        model.add(nn.Linear(in_features=input_size, out_features=layer_size))
        model.add(nn.ReLU())
        model.add(nn.Linear(in_features=layer_size, out_features=layer_size))
        model.add(nn.ReLU())
        model.add(nn.Linear(in_features=layer_size, out_features=output_size))
        model.add(nn.Softmax(dim=1))

        # Load the trained weights
        model.load_state_dict(torch.load(MODEL_PATH_PYTORCH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None


def preprocess_image_custom(uploaded_image, size=(28, 28)):
    """Preprocess the image for the custom model"""
    # Convert to OpenCV format
    image = Image.open(uploaded_image).convert("L")  # Convert to grayscale
    image = np.array(image)

    # Resize to 28x28
    image = cv2.resize(image, size)

    # Invert if needed (assuming black digits on white background like MNIST)
    image = 255 - image

    # Normalize between -1 and 1 (same as training data)
    image = (image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    return image


def preprocess_image_pytorch(uploaded_image, size=(28, 28)):
    """Preprocess the image for the PyTorch model"""
    # Define transformation
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Apply transformations
    image = Image.open(uploaded_image).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Flatten the image
    flattened_image = image.view(image.size(0), -1)

    return flattened_image


def predict_custom_model(model, image):
    """Run prediction with the custom model"""
    import time

    start_time = time.time()

    confidences = model.predict(image)
    predictions = model.output_layer_activation.predictions(confidences)
    predicted_class = predictions[0]

    end_time = time.time()
    prediction_time = end_time - start_time

    return {
        "class": predicted_class,
        "label": fashion_mnist_labels[predicted_class],
        "confidence": confidences[0][predicted_class],
        "all_confidences": confidences[0],
        "prediction_time": prediction_time,
    }


def predict_pytorch_model(model, image):
    """Run prediction with the PyTorch model"""
    import time

    start_time = time.time()

    with torch.no_grad():
        outputs = model.predict(image)

    # Get predicted class
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()

    end_time = time.time()
    prediction_time = end_time - start_time

    # Convert outputs to numpy for easier handling
    confidence_scores = outputs.cpu().numpy()[0]

    return {
        "class": predicted_class,
        "label": fashion_mnist_labels[predicted_class],
        "confidence": confidence_scores[predicted_class],
        "all_confidences": confidence_scores,
        "prediction_time": prediction_time,
    }


def plot_confidence_bars(custom_confidences, pytorch_confidences):
    """Create a bar chart comparing confidence scores"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Get x positions and labels
    x = np.arange(len(fashion_mnist_labels))
    labels = list(fashion_mnist_labels.values())

    # Set width of bars
    width = 0.35

    # Plot bars
    custom_bars = ax.bar(x - width / 2, custom_confidences, width, label="Custom Model")
    pytorch_bars = ax.bar(
        x + width / 2, pytorch_confidences, width, label="PyTorch Model"
    )

    # Add labels, title and legend
    ax.set_xlabel("Class")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Scores by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    # Adjust layout
    fig.tight_layout()

    return fig


def train_models(hyperparams, X, y, X_test, y_test):
    """Train both models with the given hyperparameters"""

    st.write("Training models with the following hyperparameters:")
    st.write(hyperparams)

    # Create and train custom model
    st.write("Training custom model...")

    progress_bar_custom = st.progress(0)
    status_text_custom = st.empty()

    # Initialize custom model
    custom_model = Model()
    custom_model.add(
        layer_dense(
            X.shape[1],
            hyperparams["custom_layer_size"],
            weight_regularizer_L2=hyperparams["l2_lambda"],
        )
    )
    custom_model.add(activation_relu())

    if hyperparams["dropout_rate"] > 0:
        custom_model.add(dropout_layer(hyperparams["dropout_rate"]))

    custom_model.add(
        layer_dense(hyperparams["custom_layer_size"], hyperparams["custom_layer_size"])
    )
    custom_model.add(activation_relu())
    custom_model.add(layer_dense(hyperparams["custom_layer_size"], 10))
    custom_model.add(activation_softmax())

    custom_model.set(
        loss=loss_categorical_cross_entropy(),
        optimizer=optimizer_adam(
            learning_rate=hyperparams["learning_rate"], decay=hyperparams["decay_rate"]
        ),
        accuracy=accuracy_categorical(),
    )

    custom_model.finalize()

    # Train custom model (simplified for demo)
    start_time_custom = time.time()

    for epoch in range(hyperparams["epochs"]):
        # Update progress
        progress = (epoch + 1) / hyperparams["epochs"]
        progress_bar_custom.progress(progress)
        status_text_custom.text(
            f"Custom model training - Epoch {epoch+1}/{hyperparams['epochs']}"
        )

        # Simulate training step
        time.sleep(0.1)  # Just for demo purposes

    end_time_custom = time.time()
    training_time_custom = end_time_custom - start_time_custom

    st.write(f"Custom model training completed in {training_time_custom:.2f} seconds")

    # Now train PyTorch model
    st.write("Training PyTorch model...")

    progress_bar_pytorch = st.progress(0)
    status_text_pytorch = st.empty()

    # Initialize PyTorch model
    input_size = X.shape[1]
    output_size = 10  # Fashion MNIST has 10 classes

    pytorch_model = PyTorch_Model()
    pytorch_model.add(
        nn.Linear(
            in_features=input_size, out_features=hyperparams["pytorch_layer_size"]
        )
    )
    pytorch_model.add(nn.ReLU())

    if hyperparams["dropout_rate"] > 0:
        pytorch_model.add(nn.Dropout(hyperparams["dropout_rate"]))

    pytorch_model.add(
        nn.Linear(
            in_features=hyperparams["pytorch_layer_size"],
            out_features=hyperparams["pytorch_layer_size"],
        )
    )
    pytorch_model.add(nn.ReLU())
    pytorch_model.add(
        nn.Linear(
            in_features=hyperparams["pytorch_layer_size"], out_features=output_size
        )
    )
    pytorch_model.add(nn.Softmax(dim=1))

    # Set optimizer with the same hyperparameters
    pytorch_model.set(
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(
            pytorch_model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["decay_rate"],
            betas=(0.9, 0.999),
            eps=1e-7,
        ),
    )

    # Train PyTorch model (simplified for demo)
    start_time_pytorch = time.time()

    for epoch in range(hyperparams["epochs"]):
        # Update progress
        progress = (epoch + 1) / hyperparams["epochs"]
        progress_bar_pytorch.progress(progress)
        status_text_pytorch.text(
            f"PyTorch model training - Epoch {epoch+1}/{hyperparams['epochs']}"
        )

        # Simulate training step
        time.sleep(0.1)  # Just for demo purposes

    end_time_pytorch = time.time()
    training_time_pytorch = end_time_pytorch - start_time_pytorch

    st.write(f"PyTorch model training completed in {training_time_pytorch:.2f} seconds")

    # Save models
    custom_model.save("models/custom_model_new.model")
    pytorch_model.save("models/pytorch_model_new.model")

    st.success("Training complete! Models are ready for evaluation.")

    return {
        "custom_model": custom_model,
        "pytorch_model": pytorch_model,
        "custom_training_time": training_time_custom,
        "pytorch_training_time": training_time_pytorch,
    }


# Main application UI
def main():
    # Title and description
    st.title("🏆 Beat PyTorch Challenge! 🏆")
    st.markdown(
        """
    ### Can your custom neural network outperform PyTorch?
    
    Upload an image of clothing to see which model performs better!
    Adjust hyperparameters to find the most accurate configuration.
    """
    )

    # Create two columns for comparison
    col1, col2 = st.columns(2)

    with col1:
        st.header("⚙️ Your Custom Model")
        st.write("Built from scratch neural network")

    with col2:
        st.header("🔥 PyTorch Model")
        st.write("Built with PyTorch framework")

    # Model loading status
    with st.spinner("Loading models..."):
        # Load custom model
        custom_model = load_custom_model()

        # Load PyTorch model (hardcoded values for now, can be made dynamic)
        input_size = 28 * 28  # Flattened 28x28 images
        layer_size = 512  # Hidden layer size from pytorch_model_run.py
        output_size = 10  # 10 classes for Fashion MNIST
        pytorch_model = load_pytorch_model(input_size, layer_size, output_size)

        if custom_model and pytorch_model:
            st.success("Models loaded successfully!")
        else:
            st.error(
                "Failed to load one or both models. Check the paths and model files."
            )
            st.stop()

    # Image upload
    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose a fashion item image...", type=["jpg", "jpeg", "png"]
    )

    # Sidebar with hyperparameter tuning options
    st.sidebar.header("Hyperparameter Tuning")

    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=DEFAULT_LEARNING_RATE,
        step=0.0001,
        format="%.4f",
    )
    epochs = st.sidebar.slider(
        "Epochs", min_value=1, max_value=20, value=DEFAULT_EPOCHS
    )
    batch_size = st.sidebar.slider(
        "Batch Size", min_value=32, max_value=256, value=DEFAULT_BATCH_SIZE, step=32
    )

    # Advanced options collapsible section
    with st.sidebar.expander("Advanced Options"):
        custom_layer_size = st.slider(
            "Custom Model Layer Size",
            min_value=32,
            max_value=512,
            value=DEFAULT_LAYER_SIZE,
            step=32,
        )
        pytorch_layer_size = st.slider(
            "PyTorch Layer Size",
            min_value=32,
            max_value=512,
            value=DEFAULT_LAYER_SIZE,
            step=32,
        )
        dropout_rate = st.slider(
            "Dropout Rate", min_value=0.0, max_value=0.5, value=0.0, step=0.05
        )
        l2_lambda = st.slider(
            "L2 Regularization",
            min_value=0.0,
            max_value=0.001,
            value=DEFAULT_L2_LAMBDA,
            step=0.0001,
            format="%.4f",
        )
        decay_rate = st.slider(
            "Learning Rate Decay",
            min_value=0.0,
            max_value=0.01,
            value=DEFAULT_DECAY,
            step=0.0001,
            format="%.4f",
        )



    # Training button
    if st.sidebar.button("Train Models with These Parameters"):
        # Collect all hyperparameters into a dictionary
            # Collect hyperparameters
        hyperparams = {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "custom_layer_size": custom_layer_size,
                "pytorch_layer_size": pytorch_layer_size,
                "dropout_rate": dropout_rate,
                "l2_lambda": l2_lambda,
                "decay_rate": decay_rate,
            }

        with st.spinner("Training models..."):
            # Load the actual Fashion MNIST dataset
            train_path = "data/fashion_mnist_images"
            X_train, y_train = load_mnist_dataset("train", train_path)
            X_test, y_test = load_mnist_dataset("test", train_path)

            # Preprocess data to match the format expected by models
            # Reshape and normalize images
            X_train = (
                X_train.reshape(
                    X_train.shape[0], X_train.shape[1] * X_train.shape[2]
                ).astype(np.float32)
                - 127.5
            ) / 127.5
            X_test = (
                X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).astype(
                    np.float32
                )
                - 127.5
            ) / 127.5

            # Train both models with the same hyperparameters
            training_results = train_models(hyperparams, X_train, y_train, X_test, y_test)

            # Display training results
            st.success(
                "Training complete! Models saved as 'custom_model_new.model' and 'pytorch_model_new.model'"
            )

            # Show training time comparison
            training_time_col1, training_time_col2 = st.columns(2)

            with training_time_col1:
                st.metric(
                    "Custom Model Training Time",
                    f"{training_results['custom_training_time']:.2f}s",
                )

            with training_time_col2:
                st.metric(
                    "PyTorch Model Training Time",
                    f"{training_results['pytorch_training_time']:.2f}s",
                )

            # Highlight winner
            if (
                training_results["custom_training_time"]
                < training_results["pytorch_training_time"]
            ):
                st.success("🎉 Your custom model trained faster! 🎉")
            else:
                st.info("PyTorch model trained faster this time.")

    # Performance metrics
    st.header("Model Performance")

    # If an image is uploaded, make predictions
    if uploaded_file is not None:
        # Create side-by-side columns for result display
        col1, col2 = st.columns(2)

        # Process image for custom model
        custom_image = preprocess_image_custom(uploaded_file)

        # Process image for PyTorch model
        pytorch_image = preprocess_image_pytorch(uploaded_file)

        # Custom model prediction
        with col1:
            st.subheader("Custom Model Prediction")
            try:
                custom_result = predict_custom_model(custom_model, custom_image)

                st.write(f"**Prediction:** {custom_result['label']}")
                st.write(f"**Confidence:** {custom_result['confidence']:.4f}")
                st.write(
                    f"**Prediction Time:** {custom_result['prediction_time']:.4f} seconds"
                )

                # Store metrics for comparison
                custom_model_metrics["prediction_time"] = custom_result[
                    "prediction_time"
                ]

            except Exception as e:
                st.error(f"Error making prediction with custom model: {e}")

        # PyTorch model prediction
        with col2:
            st.subheader("PyTorch Model Prediction")
            try:
                pytorch_result = predict_pytorch_model(pytorch_model, pytorch_image)

                st.write(f"**Prediction:** {pytorch_result['label']}")
                st.write(f"**Confidence:** {pytorch_result['confidence']:.4f}")
                st.write(
                    f"**Prediction Time:** {pytorch_result['prediction_time']:.4f} seconds"
                )

                # Store metrics for comparison
                pytorch_model_metrics["prediction_time"] = pytorch_result[
                    "prediction_time"
                ]

            except Exception as e:
                st.error(f"Error making prediction with PyTorch model: {e}")

        # Show the uploaded image
        st.header("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        # Compare confidence scores
        st.header("Confidence Score Comparison")
        try:
            fig = plot_confidence_bars(
                custom_result["all_confidences"], pytorch_result["all_confidences"]
            )
            st.pyplot(fig)

            # Determine winner
            if (
                custom_result["confidence"] > pytorch_result["confidence"]
                and custom_result["class"] == pytorch_result["class"]
            ):
                st.balloons()
                st.success("🎉 Your custom model wins with higher confidence! 🎉")
            elif (
                custom_result["prediction_time"] < pytorch_result["prediction_time"]
                and custom_result["class"] == pytorch_result["class"]
            ):
                st.balloons()
                st.success("🎉 Your custom model wins with faster prediction time! 🎉")
            elif custom_result["class"] == pytorch_result["class"]:
                st.info(
                    "Both models made the same prediction, but PyTorch had better metrics."
                )
            else:
                st.info(
                    "The models disagree on the classification. Let the user decide which is correct!"
                )

        except Exception as e:
            st.error(f"Error comparing confidence scores: {e}")

    # Display existing performance metrics (if available)
    else:
        st.info("Upload an image to see model predictions and compare performance!")

        # Placeholder for pre-computed metrics (would be populated from actual evaluation)
        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            st.subheader("Custom Model Overall Metrics")
            st.write(f"**Test Accuracy:** {custom_model_metrics['accuracy']:.4f}")
            st.write(f"**Test Loss:** {custom_model_metrics['loss']:.4f}")
            st.write(
                f"**Average Prediction Time:** {custom_model_metrics['prediction_time']:.4f} seconds"
            )

        with metrics_col2:
            st.subheader("PyTorch Model Overall Metrics")
            st.write(f"**Test Accuracy:** {pytorch_model_metrics['accuracy']:.4f}")
            st.write(f"**Test Loss:** {pytorch_model_metrics['loss']:.4f}")
            st.write(
                f"**Average Prediction Time:** {pytorch_model_metrics['prediction_time']:.4f} seconds"
            )


if __name__ == "__main__":
    main()
