
# 🎭 Emotion Detector

A machine learning–based web app that detects and classifies human emotions in real-time using a webcam feed. Built with Python, TensorFlow, OpenCV, and a React frontend.

---

## 🚀 Demo

![Demo Screenshot](demo/screenshot.png)  
*Live webcam input → Face detection → Emotion classification → Instant feedback*

---

## 🧠 Technologies Used

### 🖥️ Frontend
- React
- HTML, CSS, JavaScript
- Webpack / npm

### 🧪 Backend
- Python
- Flask
- TensorFlow / Keras
- OpenCV
- Pretrained CNN model

### 💾 Other Tools
- Git
- VS Code
- REST API
- JSON
- Webcam integration (via browser)

---

## ⚙️ Setup & Deployment

### 1. Clone the repository:
```bash
git clone https://github.com/mateuszkaron/emotion-detector.git
cd emotion-detector
```

### 2. Install frontend dependencies:
```bash
cd frontend
npm install
```

### 3. Build the frontend:
```bash
npm run build
```

### 4. Run the backend:
```bash
cd ../backend
# (optional) python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 5. (Optional) Run frontend in dev mode:
```bash
cd ../frontend
npm start
```
---

## 📌 Notes

- `frontend/build/` and `data/` are ignored by Git.
- Trained model files (`models/*.h5`) are included by default.
- If you're training the model yourself, you'll need to provide a labeled dataset in the `data/` folder.

---

## 🧪 Model Training

The backend uses a CNN trained on labeled facial emotion datasets (e.g., FER2013). You can retrain the model by modifying the training script and supplying your own dataset in the `data/` folder.

---

## 📬 Contact

Created by [Mateusz Karoń](https://www.linkedin.com/in/mateusz-karon-dev/)
