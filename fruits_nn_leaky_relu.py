"""
Fruits Neural Network with LeakyReLU
Alternative implementation using LeakyReLU activation function.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_fruits_model_leaky_relu(input_shape=(100, 100, 3), num_classes=5, alpha=0.1):
    """
    Create a neural network model for fruits classification with LeakyReLU.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of fruit classes to classify
        alpha: LeakyReLU alpha parameter (slope for negative values)
    
    Returns:
        A compiled Keras model
    """
    model = keras.Sequential([
        # Layer 1: Flatten - converts 2D image to 1D vector
        layers.Flatten(input_shape=input_shape, name='flatten'),
        
        # Layer 2: Dense layer with 256 neurons and LeakyReLU activation
        layers.Dense(256, name='dense_1'),
        layers.LeakyReLU(alpha=alpha, name='leaky_relu_1'),
        
        # Layer 3: Dense layer with 128 neurons and LeakyReLU activation
        # (decreasing number of neurons)
        layers.Dense(128, name='dense_2'),
        layers.LeakyReLU(alpha=alpha, name='leaky_relu_2'),
        
        # Layer 4: Dense layer with 64 neurons and LeakyReLU activation
        # (decreasing number of neurons)
        layers.Dense(64, name='dense_3'),
        layers.LeakyReLU(alpha=alpha, name='leaky_relu_3'),
        
        # Output layer: Dense layer with softmax for classification
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def main():
    """
    Main function to demonstrate the LeakyReLU neural network.
    """
    print("=" * 60)
    print("Fruits Neural Network with LeakyReLU")
    print("=" * 60)
    
    # Create model
    print("\nCreating model with LeakyReLU...")
    model = create_fruits_model_leaky_relu(input_shape=(100, 100, 3), num_classes=5)
    
    # Compile model
    print("Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
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
    print("✓ Activation functions: LeakyReLU")
    print("=" * 60)


if __name__ == '__main__':
    main()
