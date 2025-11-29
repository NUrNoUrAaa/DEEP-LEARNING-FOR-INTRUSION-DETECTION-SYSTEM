# Backend Deployment Guide

## Option 1: Deploy Flask to Heroku (Recommended)

### Prerequisites
- Heroku account
- Heroku CLI installed

### Steps

1. **Create Heroku app**
```bash
heroku create your-app-name
```

2. **Create Procfile**
```
web: python Model/flask_app.py
```

3. **Create runtime.txt**
```
python-3.11.6
```

4. **Push to Heroku**
```bash
git push heroku main
```

5. **Get your backend URL**
```
https://your-app-name.herokuapp.com
```

---

## Option 2: Deploy Flask to Render

### Prerequisites
- Render account
- GitHub repository connected

### Steps

1. Connect GitHub repo to Render
2. Create new Web Service
3. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python Model/flask_app.py`
   - **Environment Variables**: 
     - `FLASK_ENV=production`

4. Get your URL and update Netlify environment variables

---

## Option 3: Deploy Flask to Railway

### Steps

1. Connect to Railway
2. Set start command: `python Model/flask_app.py`
3. Railway auto-detects Python requirements

---

## Connect Backend to Netlify Frontend

### In Netlify Dashboard:

1. Go to **Settings** → **Build & Deploy** → **Environment**
2. Add new variable:
   - **Key**: `FLASK_API_URL`
   - **Value**: `https://your-backend-url.com` (Heroku/Render/Railway URL)

### Redeploy Frontend:
```bash
netlify deploy --prod
```

---

## API Endpoints

All API calls from frontend go through `/api` prefix:

- `GET /api/health` - Health check
- `POST /api/predict` - Make predictions
- `GET /api/analytics` - Get analytics
- `POST /api/train` - Train model
- `GET /api/settings` - Get settings
- `POST /api/settings` - Update settings

---

## Local Development

### Terminal 1: Backend
```bash
cd Model
source ../venv/bin/activate  # or .\venv\Scripts\activate on Windows
python flask_app.py
```

Backend runs on `http://localhost:5000`

### Terminal 2: Frontend
```bash
npm start
```

Frontend runs on `http://localhost:4200`

Angular automatically proxies `/api/*` to `localhost:5000/api/*`

---

## Production Architecture

```
┌─────────────────┐
│   Netlify CDN   │ (Frontend: dist/ids-dashboard/browser)
│  IDSwithAI      │
└────────┬────────┘
         │
         │ API calls to /api/*
         │
┌────────▼───────────────────────┐
│ Netlify Functions (api.js)     │ (Proxy layer)
└────────┬───────────────────────┘
         │
         │ Forward to FLASK_API_URL
         │
┌────────▼──────────────────────┐
│  Flask Backend                │ (Heroku/Render/Railway)
│  Model: binary_model_final    │
│  Port: 5000                   │
└───────────────────────────────┘
```

---

## Troubleshooting

### "Backend service unavailable"
- Check `FLASK_API_URL` env variable in Netlify
- Verify backend is running and accessible
- Check CORS settings in Flask

### CORS errors
- Backend has `CORS(app)` enabled
- Netlify function proxy adds CORS headers
- Check browser console for specific error

### Models not loading
- Verify `Model/` folder has:
  - `binary_model_final.keras`
  - `final_scaler.pkl`
  - `deployment_features.pkl`
- Check backend logs for model loading errors
