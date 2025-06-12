# PARAMETRY I ŚCIEŻKI
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

BATCH_SIZE   = 32
IMG_SIZE     = (48, 48)
EPOCHS       = 100
INITIAL_LR   = 1e-3
CLASS_NAMES  = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')

project_path    = os.path.dirname(os.path.abspath(__file__))
train_dir       = os.path.join(project_path, "data", "train")
validation_dir  = os.path.join(project_path, "data", "validation")
test_dir        = os.path.join(project_path, "data", "test")
models_dir      = os.path.join(project_path, "models")

for path_name, path in [("Train", train_dir), ("Validation", validation_dir), ("Test", test_dir)]:
    if not os.path.exists(path):
        print(f"Error: Ścieżka {path_name} nie istnieje: {path}")
        exit(1)

os.makedirs(models_dir, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE

# DATA AUGMENTATION

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

# WCZYTYWANIE DANYCH

def prepare_dataset(ds, shuffle=False):
    ds = ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.map(lambda x, y: (tf.image.random_brightness(x, 0.2), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_contrast(x, 0.8, 1.2), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.prefetch(buffer_size=AUTOTUNE)

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    class_names=list(CLASS_NAMES),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels="inferred",
    label_mode="int",
    class_names=list(CLASS_NAMES),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

train_ds = prepare_dataset(raw_train_ds, shuffle=True)
val_ds   = prepare_dataset(raw_val_ds, shuffle=False)

print("\nLiczność klas w zbiorze treningowym:")
class_counts = {name: 0 for name in CLASS_NAMES}
for images, labels in raw_train_ds:
    for l in labels.numpy():
        class_counts[CLASS_NAMES[l]] += 1
print(class_counts)

# BUDOWANIE MODELU

def build_model(input_shape=(48, 48, 1), num_classes=len(CLASS_NAMES)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="Emotion_CNN_V2")
    return model

model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=len(CLASS_NAMES))
model.summary()

# KOMPILACJA I CALLBACKI

model.compile(
    optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=os.path.join(models_dir, "best_emotion_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

earlystop_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# TRENING

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, reduce_lr_cb, earlystop_cb],
)

# WYKRESY

plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training/Validation Accuracy')
plt.savefig(os.path.join(models_dir, 'accuracy.png'))
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training/Validation Loss')
plt.savefig(os.path.join(models_dir, 'loss.png'))
plt.close()

# ZAPIS MODELU

model.save(os.path.join(models_dir, "final_emotion_model.h5"))
print("\nModele zostały zapisane w katalogu:", models_dir)

