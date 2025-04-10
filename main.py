# TensorFlow and tf.keras imports
import tensorflow as tf


# Helper liblaries
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Variables
BATCH_SIZE = 32
IMG_SIZE = (48, 48)

# Find path to the dataset
project_path = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(project_path, "data", "train")
test_path = os.path.join(project_path, "data", "test")

if not os.path.exists(train_path):
    print(f"Train path does not exist: {train_path}")
if not os.path.exists(test_path):
    print(f"Test path does not exist: {test_path}")

# Load the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "F:/emotion-live-detector/emotion-detector/data/train",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
),

test_ds = tf.keras.utils.image_dataset_from_directory(
    "F:/emotion-live-detector/emotion-detector/data/test",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break


cap.release()