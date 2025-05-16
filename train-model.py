# TensorFlow and tf.keras imports
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

# Helper liblaries
import os

# Variables
BATCH_SIZE = 32
IMG_SIZE = (48, 48)
EPOCHS = 50
class_names = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')

# Find path to the dataset
project_path = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(project_path, "data", "train")

if not os.path.exists(train_path):
    print(f"Train path does not exist: {train_path}")

# Load the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

# Normalize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Create the model
model = models.Sequential([
    layers.Input(shape=(48,48,1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    # The dataset to train on
    train_ds,
    # How many times to go through the dataset
    epochs=EPOCHS,
    # For simplicity, using the same data for validation
    validation_data=train_ds,  
)

# Save the model
os.makedirs("models", exist_ok=True)
model.save("models/emotion_model_v0.03.h5")

