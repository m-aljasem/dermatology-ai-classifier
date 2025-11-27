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


def build_original_model(input_shape=(250, 250, 3), num_classes=7):
    """Build original CNN model architecture from main.ipynb."""
    model = Sequential()
    
    # Layer one
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer two
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer three
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    
    # Layer four
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation("relu"))
    
    # Layer five
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    
    return model

