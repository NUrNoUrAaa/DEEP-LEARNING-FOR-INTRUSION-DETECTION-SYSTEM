# Steps to Push Project to GitHub

## Step 1: Check git status

```bash
git status
```

## Step 2: Add all files (except those in .gitignore)

```bash
git add .
```

## Step 3: Make first commit

```bash
git commit -m "initial commit: deep learning IDS dashboard with Angular frontend"
```

## Step 4: Change main branch name (if not exists)

```bash
git branch -M main
```

## Step 5: Add remote repository

```bash
git remote add origin https://github.com/NUrNoUrAaa/DEEP-LEARNING-FOR-INTRUSION-DETECTION-SYSTEM.git
```

## Step 6: Push the project

```bash
git push -u origin main
```

---

## Execute all commands at once:

```bash
git add .
git commit -m "initial commit: deep learning IDS dashboard with Angular frontend"
git branch -M main
git remote add origin https://github.com/NUrNoUrAaa/DEEP-LEARNING-FOR-INTRUSION-DETECTION-SYSTEM.git
git push -u origin main
```

---

## Notes:

✅ `.gitignore` is ready - automatically excludes:
- `node_modules/`
- `venv/`
- `Model/data/`
- `Model/__pycache__/`
- `.vscode/`

✅ Files to be uploaded:
- All project code
- `package.json` and `requirements.txt`
- `README.md` (with setup instructions)
- Project configuration

❌ Files not uploaded (auto-downloaded):
- `node_modules` → `npm install`
- `venv` → `python -m venv venv`
- Data → download manually from Kaggle

---

## After Successful Upload:

For new users who will clone the project:

```bash
# 1. Clone the project
git clone https://github.com/NUrNoUrAaa/DEEP-LEARNING-FOR-INTRUSION-DETECTION-SYSTEM.git

# 2. Install dependencies
npm install
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. Download data manually from Kaggle and place in Model/data/

# 4. Run the application
npm start
```
