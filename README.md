# DEEP LEARNING FOR INTRUSION DETECTION SYSTEM

A high-accuracy intrusion detection system using Deep Neural Networks (Deep Learning).

---

## 📋 Prerequisites

- **Node.js** (v18+)
- **Python** (v3.10+)
- **Git**

---

## 🚀 Complete Setup Steps

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/NUrNoUrAaa/DEEP-LEARNING-FOR-INTRUSION-DETECTION-SYSTEM.git
cd DEEP-LEARNING-FOR-INTRUSION-DETECTION-SYSTEM
```

---

### 2️⃣ Prepare Project Data

#### Download Training Data:
1. Go to: [CICIDS2017 Dataset](https://www.kaggle.com/datasets/dhoogla/cicids2017)
2. Download files from Kaggle
3. Place data files in the folder:
   ```
   Model/data/
   ```

---

### 3️⃣ Setup Python Environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 4️⃣ Setup Node.js Environment (for Dashboard)

```bash
npm install
```

---

### 5️⃣ Run the Application

#### Run Backend (Flask):
```bash
cd Model
python flask_app.py
```
Running on: `http://localhost:5000`

#### Run Frontend (Angular Dashboard):
In a new terminal window:
```bash
npm start
```
Running on: `http://localhost:4200`

---

## 📁 Project Structure

```
.
├── Model/                          # Python code and model
│   ├── app.py                      # Main training script
│   ├── flask_app.py                # Prediction API
│   ├── binary_model_final.keras    # Trained model
│   ├── deployment_features.txt     # Features used
│   ├── cicids2017-deep-learning.ipynb  # Analysis notebook
│   └── data/                       # (Data files - not uploaded)
│
├── src/                            # Angular code
│   ├── app/                        # Application components
│   │   ├── pages/                  # Main pages
│   │   ├── layout/                 # Layout components
│   │   ├── services/               # Services (API)
│   │   └── models/                 # Data models
│   └── main.ts
│
├── package.json                    # Node.js dependencies
├── requirements.txt                # Python dependencies
├── angular.json                    # Angular configuration
├── tsconfig.json                   # TypeScript configuration
├── tailwind.config.js              # Tailwind CSS configuration
└── postcss.config.js               # PostCSS configuration
```

---

## 🔧 Important Commands

### Run Tests (Angular):
```bash
npm test
```

### Build Project for Production:
```bash
npm run build
```

### Build Frontend Only:
```bash
ng build
```

---

## 📊 Model Features

- **High accuracy** in intrusion detection
- **Deep Learning model** trained on CICIDS2017 data
- **REST API** for real-time predictions
- **Interactive Dashboard** for displaying results

---

## ⚠️ Important Notes

### Files Not Uploaded (in `.gitignore`):

```
node_modules/          # Downloads automatically with: npm install
venv/                  # Downloads automatically with: python -m venv venv
.vscode/               # Editor personal settings
.github/workflows/     # (if exists)
Model/__pycache__/     # Python temporary files
Model/data/            # Training data (download manually)
```

---
