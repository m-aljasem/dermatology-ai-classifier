"""
Training script for skin cancer classification.
Exact code from main.ipynb adapted for script execution.
"""

import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization

import warnings
warnings.filterwarnings("ignore")


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
            os.makedirs(os.path.join(dir_path, str(dir_name)))
        except FileExistsError:
            continue


def train_model():
    """Main training function - exact code from main.ipynb."""
    
    # Load metadata
    meta_data = pd.read_csv("/kaggle/input/skin-cancer-dataset/HAM10000_metadata.csv")
    print("Metadata loaded")
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
    # Using the cancer type label for directory creation.
    dir_names = encoder.transform(encoder.classes_)
    
    # Data Sorting process.
    images_dir = r"/kaggle/input/skin-cancer-dataset/Skin Cancer/Skin Cancer"
    train_images_dir = r"/kaggle/working/train/"
    
    # Creating new directories.
    create_dirs(train_images_dir, dir_names)
    
    # Looping through each image in previous folder and
    # assigning them to the appropriate folder
    print("\nSorting images into directories...")
    for image in os.scandir(images_dir):
        try:
            # attempting to rename image (moving to new dir).
            img_name = image.name.split(".")[0]
            img_cancer_type = str(meta_data.dx_label[meta_data.image_id == img_name].item())  # Retrieving the dx_label for image.
            shutil.copy(os.path.join(images_dir, image.name), os.path.join(train_images_dir, img_cancer_type, image.name))
        except Exception as e:
            print(e)
    
    # Create validation set
    print("\nCreating validation set...")
    validation_images_dir = r"/kaggle/working/validation/"
    inds = []  # list to contain directory names of each image.
    five_percent_content = {}  # Dictionary containing cancer type and the value that is 5% of total number of that type of cancer images.
    
    # Finding out how many images of each cancer type exist
    for dir_name in os.scandir(train_images_dir):  # Iterating over all train images folders.
        for cancer_img in os.scandir(dir_name):  # Iterating over all images in all folders.
            inds.append(cancer_img.path.split("/")[4])  # Appending each images directro number to inds for counting & sorting purpose.
    
    # Calculating number of specific type images &
    # Calculating what 5% of each image type will be.
    for directory in dir_names:
        total_amt = inds.count(str(directory))
        c_type = encoder.inverse_transform([int(directory)])[0]
        print(f"There are {total_amt} images of {c_type} cancer.")
        print(f"5% of {c_type} cancer images is: {round(total_amt * 0.05, 0)}\n")
        five_percent_content[str(directory)] = round(total_amt * 0.05, 0)
    
    # Creating and populating validation set directory
    create_dirs(validation_images_dir, dir_names)
    
    # Moving 5% of each type into its respective validation folder.
    # Looping through each sub directory
    for sub_dir in os.scandir(train_images_dir):
        # Getting all images in current subdir
        images_paths = [image.path for image in os.scandir(sub_dir)]
        # Extracting 5% of images from each directory.
        for image_path in images_paths[: int(five_percent_content[str(sub_dir.name)])]:
            # Getting category for individual images
            image_category = image_path.split("/")[4]
            # creating new image path and moving old image to new destination.
            shutil.move(image_path, os.path.join(validation_images_dir, image_category, image_path.split("/")[-1]))
    
    # Data augmentation
    print("\nSetting up data generators...")
    img_size = 250  # Augmented image size.
    batch_size = 32
    
    generator = ImageDataGenerator(zoom_range=0.3,
                                   rotation_range=90,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split=0.1,)
    
    # Augmented training set
    augmented_train_data = generator.flow_from_directory(
        train_images_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        subset="training")
    
    # Un-augmented test set
    unaugmented_test_data = generator.flow_from_directory(
        train_images_dir,
        target_size=(img_size, img_size),
        batch_size=32,
        subset="validation")
    
    # Un-augmented dev set.
    unaugmented_dev_data = image_dataset_from_directory(
        validation_images_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size)
    
    # Model training
    print("\nBuilding model...")
    # Defining model architecture
    model = Sequential()
    
    # Layer one
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(250, 250, 3)))
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
    
    # # Dropout
    # model.add(Dropout(0.5))
    
    # Layer four
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation("relu"))
    
    # Layer five
    model.add(Dense(7))
    model.add(Activation("softmax"))
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    
    print("\nStarting training...")
    # fitting/training model
    history = model.fit_generator(augmented_train_data, validation_data=unaugmented_test_data, epochs=15, verbose=1)
    
    # Model Evaluation
    print("\nEvaluating model...")
    # Extracting metrics
    metrics = history.history
    
    train_loss = metrics["loss"]
    train_accuracy = metrics["accuracy"]
    
    test_loss = metrics["val_loss"]
    test_accuracy = metrics["val_accuracy"]
    
    # Visualizing metrics
    plt.figure(figsize=(13, 4))
    plt.subplot(1, 2, 1)
    plt.title("Loss.")
    plt.plot(train_loss, label="Train")
    plt.plot(test_loss, label="Test")
    plt.grid(True)
    plt.legend(loc="best")
    
    plt.subplot(1, 2, 2)
    plt.title("Accuracy.")
    plt.plot(train_accuracy, label="Train")
    plt.plot(test_accuracy, label="Test")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig("/kaggle/working/training_metrics.png")
    print("Training metrics saved to /kaggle/working/training_metrics.png")
    
    # Predictions on dev set
    result = model.predict(unaugmented_dev_data)
    
    true_class = []  # List containing true labels for each image.
    predicted_class = [predicted_label.argmax() for predicted_label in result]
    
    for file_path in unaugmented_dev_data.file_paths:  # Looping through each image file path in dev set
        true_class.append(int(file_path.split("/")[4]))  # Appending the image folder name/cancer type label to list
    
    # Understanding classification power of model on each class
    report = classification_report(true_class, predicted_class)
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    plt.title("Heatmap of devset prediction and actual label", fontsize=13)
    cm = confusion_matrix(true_class, predicted_class)
    sns.heatmap(cm, cmap="Reds", annot=True, fmt="d")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig("/kaggle/working/confusion_matrix.png")
    print("Confusion matrix saved to /kaggle/working/confusion_matrix.png")
    
    # Save model
    model.save("/kaggle/working/skin_cancer_model.h5")
    print("\nModel saved to /kaggle/working/skin_cancer_model.h5")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    train_model()
