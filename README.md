<<<<<<< HEAD
# MLflow GenAI & Model Training – Gemini QA + Industry Experiments

Evaluate **Google Gemini** QA models with **MLflow**, run **industry-style experiments** (regression, classification, cancer, prompt comparison), and view everything in the **local MLflow UI**.

## What this project does

- **LLM evaluation:** Wraps Gemini as an MLflow pyfunc, runs `mlflow.evaluate()` with QA metrics (token count, latency, toxicity), and writes **`eval.csv`**.
- **Tracking:** Logs to **DagsHub** (default) or **local `./mlruns`** when `MLFLOW_TRACKING_URI=file:./mlruns`.
- **Local UI:** Run `./run_mlflow_ui.sh` to open MLflow at **http://localhost:5001** and browse experiments, runs, models, and artifacts.
- **Extra experiments:** `run_extra_experiments.py` adds Regression (Diabetes), Classification (Digits), Hyperparameter Sweep, **Cancer (Breast + Multi-type)**, and Prompt Comparison (QA). Cancer runs log **datasets**, **tags**, **confusion matrix**, **sample CSV**, and register **cancer-classifier** in the Model Registry.
- **Rate limits:** Handles Gemini 429 with retry and backoff.

## Prerequisites

- Python 3.9+
- **Gemini API key** for `test.py` and Prompt Comparison ([Google AI Studio](https://aistudio.google.com/) → Get API key)
- (Optional) DagsHub account for remote tracking

## Installation

1. **Open the project**
   ```bash
   cd /path/to/mlflow-main
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```
   On Python 3.14, if installation fails on `pyarrow`, run:
   ```bash
   pip install "pyarrow>=22"
   pip install mlflow tiktoken google-genai python-dotenv dagshub pandas
   ```
   (Plus any other packages from `requirement.txt` as needed.)

## Configuration

### Gemini API key (for `test.py` and Prompt Comparison)

- **Recommended:** Copy `.env.example` to `.env` and set:
  ```env
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
- Or set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment.

### Optional environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_MODEL` | `gemini-2.0-flash` | Model ID |
| `GEMINI_REQUEST_DELAY` | `2.0` | Seconds between API calls (reduces 429s) |
| `GEMINI_RETRY_MAX` | `5` | Max retries on 429 |
| `MLFLOW_TRACKING_URI` | (DagsHub) | Set to `file:./mlruns` for local UI |

### DagsHub (optional)

By default (when `MLFLOW_TRACKING_URI` is not set), `test.py` uses DagsHub repo `mangalam-123/MLfLow`. To use your own repo, edit `test.py`: `dagshub.init(...)` and `mlflow.set_tracking_uri(...)`.

---

## UI workflow – how to run and view everything

The UI is **MLflow’s built-in app** (we don’t build it). We start it and point it at `./mlruns`. No login; it’s your local server.

### 1. Use local tracking (so the UI shows your runs)

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
```

### 2. Run the main Gemini QA evaluation

```bash
python test.py
```

- Creates experiment **LLM Evaluation**, registers model **gemini-qa**, writes **`eval.csv`** in the project root.

### 3. Run extra experiments (regression, classification, cancer, etc.)

```bash
python run_extra_experiments.py
```

Adds experiments:

| Experiment | Description |
|------------|-------------|
| **Regression (Diabetes)** | Ridge regression; params (alpha), metrics (MSE, R²), logged model |
| **Classification (Digits)** | Logistic regression on digits; accuracy, F1, logged model |
| **Hyperparameter Sweep** | Ridge alpha sweep; CV MSE |
| **Cancer (Breast + Multi-type)** | Real breast cancer + synthetic 5-type; datasets, confusion matrix, sample CSV, tags, **cancer-classifier** in Model Registry |
| **Prompt Comparison (QA)** | Same questions, different prompts; latency/tokens (needs Gemini API key) |

### 4. Start the MLflow UI

```bash
./run_mlflow_ui.sh
```

Or:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```

### 5. Open the UI

In your browser go to **http://localhost:5001** (port 5001 is used because 5000 is often taken on macOS).

In the UI you can:

- **Experiments:** Open each experiment, compare runs, view metrics and parameters.
- **Runs:** Per run – Overview (description, datasets, tags, registered models), Model metrics, Artifacts (logged model, confusion matrix, sample CSV for cancer runs).
- **Model registry:** See **gemini-qa** and **cancer-classifier** and their versions.

---

## Outputs

| Output | Where / what |
|--------|----------------|
| **Console** | Aggregated metrics and table preview from `test.py` |
| **`eval.csv`** | Per-row QA results (inputs, ground truth, outputs, latency, token_count, etc.) from `test.py` |
| **`./mlruns`** | All experiments, runs, and artifacts when using local tracking |
| **MLflow UI** | Experiments, runs, metrics, parameters, tags, datasets, artifacts, Model Registry at http://localhost:5001 |

---

## Cancer experiment – what’s logged for workflow use

For **Cancer (Breast + Multi-type)** runs we log:

- **Datasets:** Training data via `mlflow.log_input()` so the run’s “Datasets” section is populated.
- **Tags:** `project`, `data_source`, `model_type`, `n_classes` for filtering.
- **Run description:** Short note describing the run.
- **Metrics:** accuracy, f1_weighted, precision_weighted, recall_weighted, training_duration_sec.
- **Artifacts:** Confusion matrix image (`artifacts/confusion_matrix.png`), sample test CSV (`artifacts/*.csv`), plus the logged model.
- **Model Registry:** **cancer-classifier** (selected runs registered for deployment workflow).

---

## Project layout

```
mlflow-main/
├── README.md                  # This file
├── requirement.txt            # Python dependencies
├── .env.example               # Copy to .env and add GEMINI_API_KEY
├── run_mlflow_ui.sh           # Start MLflow UI on port 5001 (backend: ./mlruns)
├── test.py                    # Gemini QA evaluation → eval.csv, gemini-qa
├── run_extra_experiments.py   # Regression, Classification, Cancer, Sweep, Prompt Comparison
├── TESTING.md                 # Testing steps and workflow detail
├── eval.csv                   # Generated by test.py (per-row QA results)
├── .env                       # Your API key (create from .env.example)
└── mlruns/                    # Local tracking data (when using file:./mlruns)
```

---

## Rate limits and errors

- **429 (Gemini):** Retries with backoff. If it still fails, wait for quota reset or increase `GEMINI_RETRY_MAX` / `GEMINI_REQUEST_DELAY`.
- **404 (model):** Use a supported `GEMINI_MODEL` (e.g. `gemini-2.0-flash`, `gemini-2.5-flash`).
- **API key:** Ensure `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set for `test.py` and for Prompt Comparison in `run_extra_experiments.py`.

## License

See the repository for license information.
=======
# mlflow
Created for Shankar's assignment
>>>>>>> 36b6d46932ecfc06b26a84d97308f181c0db2b06
