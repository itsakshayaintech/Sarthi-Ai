# Sarthi-AI

Local setup instructions for Windows teammates.

## Share the project

Use a private GitHub repository or a ZIP archive. Include the source folders, `requirements.txt`, and all four files in `models/`:

- `context_model.pkl`
- `indicator_model.pkl`
- `text_model.pkl`
- `voice_model.pkl`

The model files are required to run the app. Teammates do not need to train them. Do not share `.venv/` or `database/sahayak.sqlite3`; each person gets their own environment and database. The database is created automatically when the app starts.

## Run on Windows

Install Python 3.13, then open PowerShell in the project folder and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 5001
```

Open <http://127.0.0.1:5001>, create an account, and sign in. The first dependency installation needs an internet connection. Voice transcription also needs internet access because audio speech is sent to Google's recognition service.

If port `5001` is already in use, choose another port in the Flask command and open that matching local URL.

## Sharing with Git

From the project folder, initialize and commit the files:

```powershell
git init
git add .
git commit -m "Share Sarthi-AI project"
```

Create a **private** GitHub repository and push this project to it, then invite teammates to the repository. Check that the `models/` files are included and `database/sahayak.sqlite3` and `.venv/` are not included before sharing.