# Deep Learning IDS Dashboard

Complete project with Angular frontend and Python ML backend for intrusion detection.

## Setup Instructions

### Frontend (Angular)
```bash
npm install
npm start          # Development server
npm run build      # Production build
npm test          # Run tests
```

### Backend (Python)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python Model/flask_app.py
```

## Project Structure

```
ids-dashboard/
├── src/                    # Angular frontend source
├── Model/                  # Python ML model & Flask backend
│   ├── binary_model_final.keras
│   ├── flask_app.py
│   └── data/              # Training datasets
├── dist/                   # Compiled Angular output
├── node_modules/          # npm dependencies (local only)
├── venv/                  # Python virtual environment (local only)
├── package.json           # Node dependencies
├── requirements.txt       # Python dependencies
└── .gitignore            # Excludes node_modules & venv
```

## Important Notes

- **node_modules**: Not tracked in git, run `npm install` after cloning
- **venv**: Not tracked in git, create with `python -m venv venv`
- **dist/**: Committed for easy deployment
- **data/**: All training data included in repository

## Tech Stack

- **Frontend**: Angular 20.x, TypeScript, Tailwind CSS
- **Backend**: Flask 3.0, Python 3.11
- **ML**: TensorFlow 2.14, scikit-learn, XGBoost
- **Deployment**: GitHub Actions CI/CD
