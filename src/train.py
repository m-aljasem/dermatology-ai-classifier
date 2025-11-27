"""
Training script for skin cancer classification.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from model import build_skin_cancer_model


def create_generators(df_train, df_val, df_test, image_size=(224, 224), batch_size=32):
    """Create data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1/255.0)
    
    # Adjust based on your data structure
    return None, None, None


def train_model(epochs=50, batch_size=32):
    """Main training function."""
    print("Training skin cancer classification model...")
    
    # Load data
    # df = pd.read_csv('../data/HAM10000_metadata.csv')
    # df_train, df_temp = train_test_split(df, test_size=0.2, stratify=df['dx_label'])
    # df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['dx_label'])
    
    # Create generators
    # train_gen, val_gen, test_gen = create_generators(df_train, df_val, df_test)
    
    # Build model
    model = build_skin_cancer_model(input_shape=(224, 224, 3), num_classes=7)
    
    # Ensure models directory exists
    models_dir = Path('../models')
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / 'skin_cancer_model.h5'
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(str(weights_path), save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    # Train
    # history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    # print(f"✓ Best model weights saved to {weights_path}")
    
    print("Training complete! (Enable training code and generators when data is ready.)")


if __name__ == '__main__':
    train_model()

