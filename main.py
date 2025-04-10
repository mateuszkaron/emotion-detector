# TensorFlow and tf.keras imports
import tenosrFLow as tf


# Helper liblaries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

train_ds = tf.keras.uils.image_dataset_from_directory(
    "data/train",
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
),

test_ds = tf.keras.uils.image_dataset_from_directory(
    "data/test",
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
)


#TEST
