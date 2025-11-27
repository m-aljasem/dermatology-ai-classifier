"""CNN model for skin cancer classification"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam


def build_skin_cancer_model(input_shape=(224, 224, 3), num_classes=7):
    """Build CNN for multi-class skin cancer classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_skin_lesion_cnn(input_shape=(250, 250, 3), num_classes=7):
    """
    Build a custom CNN for skin lesion classification.
    
    Architecture matches the notebook implementation.
    
    Args:
        input_shape: Tuple specifying input image dimensions (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Uncompiled Keras Sequential model
    """
    model = Sequential(name="SkinLesionCNN")
    
    # Input normalization layer
    model.add(BatchNormalization(input_shape=input_shape, name='input_bn'))
    
    # Convolutional Block 1: Extract low-level features
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', name='conv1'))
    model.add(Activation("relu", name='conv1_relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='conv1_pool'))
    
    # Convolutional Block 2: Extract mid-level features
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', name='conv2'))
    model.add(Activation("relu", name='conv2_relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='conv2_pool'))
    
    # Flatten convolutional features
    model.add(Flatten(name='flatten'))
    
    # Dense Block 1: High-level feature processing
    model.add(Dense(64, name='dense1'))
    model.add(Activation("relu", name='dense1_relu'))
    
    # Dense Block 2: Final feature refinement
    model.add(Dense(32, name='dense2'))
    model.add(Activation("relu", name='dense2_relu'))
    
    # Output layer: Multi-class classification
    model.add(Dense(num_classes, name='output'))
    model.add(Activation("softmax", name='output_softmax'))
    
    return model


def build_original_model(input_shape=(250, 250, 3), num_classes=7):
    """
    Alias for build_skin_lesion_cnn for backward compatibility.
    Use build_skin_lesion_cnn instead.
    """
    return build_skin_lesion_cnn(input_shape=input_shape, num_classes=num_classes)

