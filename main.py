import tensorflow as tf
import os
import time 
import cv2 as cv
import numpy as np

# Zmienne
IMG_SIZE = (48, 48)
MODEL_PATH = "models/emotion_model_v0.03.h5"
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
predicted_emotion = ''
emotion_timestamp = 0
display_duration = 3 
last_face_coords = None

# wczytanie modelu z MODEL_PATH
model = tf.keras.models.load_model(MODEL_PATH)

# filtr na obraz
def apply_old_tv_filter(frame):
    noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
    frame = cv.addWeighted(frame, 0.9, noise, 0.1, 0)

    for i in range(0, frame.shape[0], 2):
        frame[i, :] = frame[i, :] * 0.8

    b, g, r = cv.split(frame)
    b = np.roll(b, 1, axis=1)
    r = np.roll(r, -1, axis=1)
    frame = cv.merge([b, g, r])

    frame = cv.GaussianBlur(frame, (3, 3), 0)
    return frame

# start kamery i detekcja twarzy
cap = cv.VideoCapture(0)
facecasc = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # kopia klatki z filtrem TV
    filtered_frame = apply_old_tv_filter(frame.copy())

    # konwersja do odcieni szarości i detekcja twarzy
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    key = cv.waitKey(1) & 0xFF

    # wciś 'e' coby zobaczyć emocje (dla kazdej twarzy na klatce)
    if key == ord('e'):
        for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                if roi_gray.size == 0:
                    continue
                roi_resized = cv.resize(roi_gray, IMG_SIZE)
                input_image = np.expand_dims(np.expand_dims(roi_resized, -1), 0)
                prediction = model.predict(input_image, verbose=0)
                maxindex = int(np.argmax(prediction))
                predicted_emotion = emotion_dict[maxindex]
                emotion_timestamp = time.time()
                last_face_coords = (x, y, w, h)
                break 

    # rysowanie prostokątów wokół wykrytych twarzy
    for (x, y, w, h) in faces:
        cv.rectangle(filtered_frame, (x, y-50), (x + w, y + h + 10), (255, 0, 0), 2)

    # rysowanie nazwy emocji nad twarza jesli jest wykryta i nie minęło 3 sekundy
    # od ostatniego wykrycia (potem zniknie)
    if predicted_emotion and (time.time() - emotion_timestamp < display_duration) and last_face_coords:
        x, y, w, h = last_face_coords
        cv.putText(filtered_frame, predicted_emotion, (x + 20, y - 60),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv.imshow('Live Emotion Recognition', filtered_frame)

    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
