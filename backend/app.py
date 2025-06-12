from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2 as cv
import io
import os
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # adres frontu

# Parametry i modele
IMG_SIZE = (48, 48)
MODEL_PATH = "models/best_emotion_model.h5"
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Wczytaj model
model = tf.keras.models.load_model(MODEL_PATH)

# Klasyfikator twarzy
facecasc = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# API endpoint
@app.route('/predict', methods=['POST'])
def predict_emotion():
    print("Received files:", request.files)
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('RGB')
        image = np.array(image)
        
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        gray = cv.equalizeHist(gray)  # popraw kontrast

        faces = facecasc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400

        # Wybierz największą twarz
        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv.resize(roi_gray, IMG_SIZE)
        input_image = np.expand_dims(roi_resized, axis=-1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32') / 255.0  # normalizacja

        prediction = model.predict(input_image, verbose=0)
        max_index = int(np.argmax(prediction))
        emotion = emotion_dict[max_index]
        confidence = float(np.max(prediction)) * 100
        # Dodaj: zwracaj wszystkie pewności
        all_confidences = {emotion_dict[i]: float(prediction[0][i]) * 100 for i in range(len(emotion_dict))}
        print("Prediction vector:", prediction)
        print("Max index:", max_index)
        print("Detected emotion:", emotion)
        print("Confidence:", confidence)

        return jsonify({'emotion': emotion, 'confidence': confidence, 'all_confidences': all_confidences})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Emotion Scanner API is working!"
    
# Start serwera
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

