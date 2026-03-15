# Push this folder to GitHub

Target repo: **https://github.com/ni33waghtalentica/mlflow.git**

Run these commands in **Terminal** from this folder (`mlflow-main` on your Desktop).

## 1. Go to the project folder

```bash
cd /Users/nitinw/Desktop/mlflow-main
```

## 2. If this is not yet a git repo

```bash
git init
git remote add origin https://github.com/ni33waghtalentica/mlflow.git
```

(If you already ran this once, skip to step 3.)

## 3. Add all files, commit, and push

```bash
git add -A
git status
git commit -m "Add MLflow GenAI + industry experiments (QA, Cancer, Regression, etc.)"
git branch -M main
git push -u origin main
```

If the GitHub repo already has a README or other content, use:

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## What gets pushed

- **Included:** `.gitignore`, `README.md`, `test.py`, `run_extra_experiments.py`, `run_mlflow_ui.sh`, `requirement.txt`, `.env.example`, `TESTING.md`, `PUSH_TO_GITHUB.md`, screenshots, etc.
- **Excluded by .gitignore:** `.env` (secrets), `.venv/`, `mlruns/`, `eval.csv`, `__pycache__/`

Do not remove `.gitignore`; it keeps secrets and large/local data out of the repo.
