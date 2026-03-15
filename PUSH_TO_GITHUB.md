# Push this folder to GitHub

Target repo: **https://github.com/ni33waghtalentica/mlflow.git**

An initial push (`.gitignore`, this file, and merge with the repo’s README) is already done. To add **all remaining files** (e.g. `test.py`, `run_extra_experiments.py`, full `README.md`, etc.) from your Desktop `mlflow-main` folder, run the following in **Terminal**.

## 1. Go to the project folder

```bash
cd /Users/nitinw/Desktop/mlflow-main
```

## 2. If this folder is not yet a git repo

```bash
git init
git remote add origin https://github.com/ni33waghtalentica/mlflow.git
git pull origin main --allow-unrelated-histories --no-rebase --no-edit
```

(If the repo is already set up and you only want to add more files, skip to step 3.)

## 3. Add all files, commit, and push

```bash
git add -A
git status
git commit -m "Add MLflow GenAI + industry experiments (QA, Cancer, Regression, etc.)"
git push origin main
```

## What gets pushed

- **Included:** `.gitignore`, `README.md`, `test.py`, `run_extra_experiments.py`, `run_mlflow_ui.sh`, `requirement.txt`, `.env.example`, `TESTING.md`, `PUSH_TO_GITHUB.md`, screenshots, etc.
- **Excluded by .gitignore:** `.env` (secrets), `.venv/`, `mlruns/`, `eval.csv`, `__pycache__/`

Do not remove `.gitignore`; it keeps secrets and large/local data out of the repo.
