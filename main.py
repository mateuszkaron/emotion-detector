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
model = tf.keras.models.load_model("models/emotion_model.h5")
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
predicted_class = ''

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
    train_path,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
),

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
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

    # Convert the frame to grayscale and resize it
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, IMG_SIZE)
    normalized = resized / 255.0
    input_image = normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    display_frame = frame.copy()

    # Update emotion with 'e'
    key = cv.waitKey(1) & 0xFF
    if key == ord('e'):
        pred = model.predict(input_image)
        predicted_class = class_names[np.argmax(pred)]
    
    if predicted_class:
        cv.putText(display_frame, f"Emocja: {predicted_class}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    
    cv.imshow('Live Emotion Recognition', display_frame)
        
    # Stop with 'q'
    if key == ord('q'):
        break

cap.release() 