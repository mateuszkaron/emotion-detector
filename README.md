# Emotion Detector - Deployment & Development Instructions

## After cloning the repository, follow these steps:

1. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```
2. **Build the frontend:**
   ```bash
   npm run build
   ```
3. **Run the backend:**
   ```bash
   cd ../backend
   # (opcjonalnie) python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```
4. **Run the frontend in development mode:**
   ```bash
   cd ../frontend
   npm start
   ```

---

- The `data/` folder (with images) and `frontend/build/` are ignored by Git.
- After building the frontend  (`npm run build`), the production files will appear in `frontend/build/`.
- Always run the backend from the `backend/` directory.
- Model files (`models/*.h5`) are tracked by Git by default (you can add them to `.gitignore` if you want to exclude them).

**Note:**
- Before running the backend for the first time, make sure to install the required Python libraries (`pip install -r requirements.txt`).
- If you want to train the model, you need to manually create and populate the `data/` folder.
