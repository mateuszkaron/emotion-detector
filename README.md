# Emotion Detector - Deployment & Development Instructions

## Po klonowaniu repozytorium wykonaj:

1. **Zainstaluj zależności frontendu:**
   ```bash
   cd frontend
   npm install
   ```
2. **Zbuduj frontend:**
   ```bash
   npm run build
   ```
3. **Uruchom backend:**
   ```bash
   cd ../backend
   # (opcjonalnie) python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```
4. **Uruchom frontend (dev):**
   ```bash
   cd ../frontend
   npm start
   ```

---

- Folder `data/` (ze zdjęciami) oraz `frontend/build/` są ignorowane przez git.
- Po zbudowaniu frontendu (`npm run build`) pliki produkcyjne pojawią się w `frontend/build/`.
- Backend uruchamiaj z katalogu `backend/`.
- Pliki modeli (`models/*.h5`) są domyślnie śledzone przez git (możesz dodać je do `.gitignore`, jeśli chcesz je pominąć).

**Uwaga:**
- Przed pierwszym uruchomieniem backendu zainstaluj wymagane biblioteki Pythona (`pip install -r requirements.txt`).
- Jeśli chcesz trenować model, folder `data/` musisz utworzyć i uzupełnić samodzielnie.
