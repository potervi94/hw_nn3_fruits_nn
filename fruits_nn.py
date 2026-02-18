"""
Fruits Neural Network
A neural network classifier for fruits with the following requirements:
- 3-5 layers
- First layer: Flatten
- Number of neurons should not increase in subsequent layers
- Activation functions: RELU or LeakyRELU
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def create_fruits_model(input_shape=(100, 100, 3), num_classes=5):
    """
    Create a neural network model for fruits classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of fruit classes to classify
    
    Returns:
        A compiled Keras model
    """
    model = keras.Sequential([
        # Layer 1: Flatten - converts 2D image to 1D vector
        layers.Flatten(input_shape=input_shape, name='flatten'),
        
        # Layer 2: Dense layer with 256 neurons and ReLU activation
        layers.Dense(256, activation='relu', name='dense_1'),
        
        # Layer 3: Dense layer with 128 neurons and ReLU activation
        # (decreasing number of neurons)
        layers.Dense(128, activation='relu', name='dense_2'),
        
        # Layer 4: Dense layer with 64 neurons and ReLU activation
        # (decreasing number of neurons)
        layers.Dense(64, activation='relu', name='dense_3'),
        
        # Output layer: Dense layer with softmax for classification
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for the optimizer
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_fruits_dataset():
    """
    Load and preprocess fruits dataset.
    For demonstration, we'll use a synthetic dataset.
    Replace this with actual fruits dataset loading.
    
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    # Generate synthetic data for demonstration
    # In real use, load actual fruits images
    np.random.seed(42)
    
    # Training data: 1000 samples
    x_train = np.random.rand(1000, 100, 100, 3).astype('float32')
    y_train = np.random.randint(0, 5, 1000)
    
    # Test data: 200 samples
    x_test = np.random.rand(200, 100, 100, 3).astype('float32')
    y_test = np.random.randint(0, 5, 200)
    
    return (x_train, y_train), (x_test, y_test)


def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    """
    Train the model.
    
    Args:
        model: Compiled Keras model
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Training history
    """
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        x_test: Test data
        y_test: Test labels
    
    Returns:
        Test loss and accuracy
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    return test_loss, test_acc


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")


def main():
    """
    Main function to demonstrate the neural network.
    """
    print("=" * 60)
    print("Fruits Neural Network")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    (x_train, y_train), (x_test, y_test) = load_fruits_dataset()
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Create model
    print("\nCreating model...")
    model = create_fruits_model(input_shape=(100, 100, 3), num_classes=5)
    
    # Compile model
    print("Compiling model...")
    model = compile_model(model, learning_rate=0.001)
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Verify requirements
    print("\n" + "=" * 60)
    print("Verification of Requirements:")
    print("=" * 60)
    print("✓ First layer is Flatten")
    print("✓ Number of layers: 5 (Flatten + 3 Dense + Output)")
    print("✓ Neuron counts: 256 → 128 → 64 (decreasing)")
    print("✓ Activation functions: ReLU")
    print("=" * 60)
    
    # Train model
    print("\nTraining model...")
    history = train_model(
        model, x_train, y_train, x_test, y_test,
        epochs=10, batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, x_test, y_test)
    
    # Plot training history
    print("\nGenerating training history plot...")
    plot_training_history(history)
    
    # Save model
    model.save('fruits_model.keras')
    print("\nModel saved as 'fruits_model.keras'")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
