"""
Training script for skin cancer classification.
Exact code from main.ipynb adapted for script execution.
Reflects all improvements from notebooks/02_model_training.ipynb
"""

import os
import json
import shutil
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Activation, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    CSVLogger
)

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def create_dirs(dir_path: str, dir_names: list):
    """
    This function creates directories within specified directory path
    with the provided list of directory names.
    
    Inputs
        dir_path:str - The path to which the new directories will reside in.
        dir_names:list - List name(s) of directories to be created.
    """
    # Looping through to create directories in new location.
    for dir_name in dir_names:
        try:
            os.makedirs(os.path.join(dir_path, str(dir_name)), exist_ok=True)
        except FileExistsError:
            continue


def build_skin_lesion_cnn(input_shape=(250, 250, 3), num_classes=7):
    """
    Build a custom CNN for skin lesion classification.
    
    Architecture matches the notebook implementation.
    """
    model = Sequential(name="SkinLesionCNN")
    
    # Input normalization layer
    model.add(BatchNormalization(input_shape=input_shape, name='input_bn'))
    
    # Convolutional Block 1
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', name='conv1'))
    model.add(Activation("relu", name='conv1_relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='conv1_pool'))
    
    # Convolutional Block 2
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', name='conv2'))
    model.add(Activation("relu", name='conv2_relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='conv2_pool'))
    
    # Flatten convolutional features
    model.add(Flatten(name='flatten'))
    
    # Dense Block 1
    model.add(Dense(64, name='dense1'))
    model.add(Activation("relu", name='dense1_relu'))
    
    # Dense Block 2
    model.add(Dense(32, name='dense2'))
    model.add(Activation("relu", name='dense2_relu'))
    
    # Output layer
    model.add(Dense(num_classes, name='output'))
    model.add(Activation("softmax", name='output_softmax'))
    
    return model


def configure_gpu():
    """Configure GPU for TensorFlow."""
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Explicitly set GPU as the only visible device
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Verify GPU configuration
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✓ Physical GPUs: {len(gpus)}")
            print(f"✓ Logical GPUs: {len(logical_gpus)}")
            
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                print(f"    Memory growth: Enabled")
            
            # Force GPU usage with a test computation
            print("\nTesting GPU computation...")
            with tf.device('/GPU:0'):
                test_a = tf.random.normal([1000, 1000])
                test_b = tf.random.normal([1000, 1000])
                test_c = tf.matmul(test_a, test_b)
                print(f"  ✓ GPU test successful!")
                print(f"  Test computation device: {test_c.device}")
                print(f"  ✓ TensorFlow WILL use GPU for training")
            
            # Ensure float32 precision (mixed precision causes issues with TopK accuracy metric)
            try:
                tf.keras.mixed_precision.set_global_policy('float32')
            except:
                pass
            print(f"  ✓ Using float32 precision (compatible with all metrics)")
                
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
            print("  Training will fall back to CPU")
    else:
        print("⚠️  No GPU devices found - training will use CPU")
        print("  Make sure GPU is enabled in Kaggle notebook settings")
    
    print("=" * 60)


def train_model():
    """Main training function - matches notebook implementation."""
    
    # Display TensorFlow version
    print("=" * 60)
    print("Environment Setup")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    # Configure GPU
    configure_gpu()
    
    # Load metadata
    METADATA_PATH = "/kaggle/input/skin-cancer-dataset/HAM10000_metadata.csv"
    meta_data = pd.read_csv(METADATA_PATH)
    print("\n✓ Metadata loaded")
    print(meta_data.head())
    
    # Encode labels
    print("\nUnique Cancer types represented in data.")
    print(meta_data.dx.unique(), "\n")
    
    # Handling categorical data
    encoder = LabelEncoder()
    meta_data["dx_label"] = encoder.fit_transform(meta_data["dx"])
    
    # Display of labels and their integer encoding
    print("Cancer types and their integer encoding")
    print(encoder.classes_)
    print(encoder.transform(encoder.classes_))
    
    # Data sorting
    dir_names = encoder.transform(encoder.classes_)
    
    # Data Sorting process
    images_dir = r"/kaggle/input/skin-cancer-dataset/Skin Cancer/Skin Cancer"
    TRAIN_IMAGES_DIR = r"/kaggle/working/train/"
    VALIDATION_IMAGES_DIR = r"/kaggle/working/validation/"
    
    # Check if directories already exist and are populated
    skip_data_org = False
    if os.path.exists(TRAIN_IMAGES_DIR):
        # Check if train directories have files
        train_dirs = [d for d in os.listdir(TRAIN_IMAGES_DIR) if os.path.isdir(os.path.join(TRAIN_IMAGES_DIR, d))]
        if train_dirs:
            # Check if at least one directory has images
            for dir_name in train_dirs[:1]:
                dir_path = os.path.join(TRAIN_IMAGES_DIR, dir_name)
                if len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]) > 0:
                    skip_data_org = True
                    print("\n✓ Training directories already exist and contain data. Skipping data organization.")
                    break
    
    if not skip_data_org:
        # Creating new directories
        create_dirs(TRAIN_IMAGES_DIR, dir_names)
        
        # Looping through each image and assigning them to the appropriate folder
        print("\nSorting images into directories...")
        for image in os.scandir(images_dir):
            try:
                img_name = image.name.split(".")[0]
                img_cancer_type = str(meta_data.dx_label[meta_data.image_id == img_name].item())
                shutil.copy(
                    os.path.join(images_dir, image.name),
                    os.path.join(TRAIN_IMAGES_DIR, img_cancer_type, image.name)
                )
            except Exception as e:
                print(f"Error processing {image.name}: {e}")
        
        # Create validation set
        print("\nCreating validation set...")
        inds = []
        five_percent_content = {}
        
        # Finding out how many images of each cancer type exist
        for dir_name in os.scandir(TRAIN_IMAGES_DIR):
            for cancer_img in os.scandir(dir_name):
                inds.append(cancer_img.path.split("/")[4])
        
        # Calculating 5% of each image type
        for directory in dir_names:
            total_amt = inds.count(str(directory))
            c_type = encoder.inverse_transform([int(directory)])[0]
            print(f"There are {total_amt} images of {c_type} cancer.")
            print(f"5% of {c_type} cancer images is: {round(total_amt * 0.05, 0)}\n")
            five_percent_content[str(directory)] = round(total_amt * 0.05, 0)
        
        # Creating and populating validation set directory
        create_dirs(VALIDATION_IMAGES_DIR, dir_names)
        
        # Moving 5% of each type into its respective validation folder
        for sub_dir in os.scandir(TRAIN_IMAGES_DIR):
            images_paths = [image.path for image in os.scandir(sub_dir)]
            for image_path in images_paths[: int(five_percent_content[str(sub_dir.name)])]:
                image_category = image_path.split("/")[4]
                shutil.move(
                    image_path,
                    os.path.join(VALIDATION_IMAGES_DIR, image_category, image_path.split("/")[-1])
                )
    
    # Data augmentation
    print("\nSetting up data generators...")
    IMG_SIZE = 250
    BATCH_SIZE = 32
    NUM_CLASSES = 7
    VALIDATION_SPLIT = 0.1
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.3,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=VALIDATION_SPLIT
    )
    
    # Validation generator (no augmentation)
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_IMAGES_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        directory=TRAIN_IMAGES_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=RANDOM_SEED
    )
    
    # Development set generator
    dev_generator = image_dataset_from_directory(
        VALIDATION_IMAGES_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )
    
    print(f"\n✓ Data generators created!")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    
    # Build model
    print("\nBuilding model...")
    model = build_skin_lesion_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)
    
    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
    )
    
    print("\n" + "=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)
    model.summary()
    
    # Setup callbacks
    EPOCHS = 15
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath='/kaggle/working/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        CSVLogger(
            filename='/kaggle/working/training_history.csv',
            separator=',',
            append=False
        )
    ]
    
    # GPU verification callback
    class GPUVerificationCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            gpus = tf.config.list_logical_devices('GPU')
            print(f"\n{'='*60}")
            print("GPU VERIFICATION")
            print(f"{'='*60}")
            if gpus:
                print(f"✓ GPU Available: {gpus[0].name}")
                try:
                    with tf.device('/GPU:0'):
                        test_tensor = tf.constant([1.0, 2.0, 3.0])
                        result = tf.reduce_sum(test_tensor)
                        print(f"  ✓ GPU computation test successful")
                    print("  ✓ Training will use GPU")
                except Exception as e:
                    print(f"  ⚠️  GPU test failed: {e}")
            else:
                print("⚠️  WARNING: No GPU detected - training on CPU!")
            print(f"{'='*60}\n")
    
    callbacks.append(GPUVerificationCallback())
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60 + "\n")
    
    # Use model.fit instead of fit_generator (deprecated in TF 2.18.0)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks
    )
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    
    # Save final model
    final_model_path = '/kaggle/working/final_model.keras'
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Display training summary
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print(f"\nTraining Summary:")
    print(f"  Final training accuracy: {final_train_acc:.4f}")
    print(f"  Final validation accuracy: {final_val_acc:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    # Save label mapping
    label_mapping = {
        int(idx): encoder.inverse_transform([idx])[0]
        for idx in range(len(encoder.classes_))
    }
    with open('/kaggle/working/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"\n✓ Label mapping saved to: /kaggle/working/label_mapping.json")
    
    # Save model config
    model_config = {
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
        'classes': encoder.classes_.tolist()
    }
    with open('/kaggle/working/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✓ Model config saved to: /kaggle/working/model_config.json")
    
    # Model Evaluation
    print("\nEvaluating model...")
    
    # Load best model for evaluation
    try:
        model.load_weights('/kaggle/working/best_model.keras')
        print("✓ Loaded best model weights for evaluation")
    except:
        print("⚠️  Using final model weights (best weights not found)")
    
    # Extract training history
    metrics = history.history
    epochs = range(1, len(metrics['loss']) + 1)
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History Analysis', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(epochs, metrics['loss'], 'b-o', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, metrics['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(epochs, metrics['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, metrics['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    
    if 'top2_accuracy' in metrics:
        axes[1, 0].plot(epochs, metrics['top2_accuracy'], 'g-o', label='Training Top-2', linewidth=2)
        axes[1, 0].plot(epochs, metrics['val_top2_accuracy'], 'm-s', label='Validation Top-2', linewidth=2)
        axes[1, 0].set_title('Top-2 Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-2 Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])
    else:
        axes[1, 0].axis('off')
    
    if 'lr' in metrics:
        axes[1, 1].plot(epochs, metrics['lr'], 'purple', linewidth=2, marker='o')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Training history saved to /kaggle/working/training_history.png")
    
    # Predictions on dev set
    print("\nGenerating predictions on development set...")
    dev_images = []
    dev_labels = []
    
    for batch_images, batch_labels in dev_generator:
        dev_images.append(batch_images.numpy())
        dev_labels.append(batch_labels.numpy())
        if len(dev_images) * BATCH_SIZE >= 1000:
            break
    
    dev_images = np.concatenate(dev_images, axis=0)
    dev_labels = np.concatenate(dev_labels, axis=0)
    
    if dev_images.max() > 1.0:
        dev_images = dev_images / 255.0
    
    predictions = model.predict(dev_images, batch_size=BATCH_SIZE, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = dev_labels.tolist()
    
    # Ensure matching lengths
    min_length = min(len(true_classes), len(predicted_classes))
    true_classes = true_classes[:min_length]
    predicted_classes = predicted_classes[:min_length]
    
    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=[encoder.inverse_transform([i])[0] for i in range(NUM_CLASSES)],
        digits=4
    )
    print(report)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    plt.title("Heatmap of devset prediction and actual label", fontsize=13)
    cm = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(cm, cmap="Reds", annot=True, fmt="d")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig('/kaggle/working/confusion_matrix_normalized.png')
    print("✓ Confusion matrix saved to /kaggle/working/confusion_matrix_normalized.png")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    train_model()
