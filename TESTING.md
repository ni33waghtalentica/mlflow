# Testing Guide & Workflow

## Verification (checked at this end)

- **MLflow UI**: Responding with HTTP 200 at http://localhost:5001
- **Backend store**: `./mlruns` contains experiment **"LLM Evaluation"**, one run, and registered model **gemini-qa**
- **Script**: `test.py` runs correctly; it only stops during evaluation when no Gemini API key is set

---

## Steps to Perform to Test

### 1. One-time setup

```bash
cd /Users/nitinw/Desktop/mlflow-main
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install "pyarrow>=22"  # if Python 3.14
pip install mlflow tiktoken google-genai python-dotenv dagshub pandas
```

Or, if your Python is 3.9–3.12:

```bash
pip install -r requirement.txt
```

### 2. Add your Gemini API key

```bash
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_actual_key
# Get key: https://aistudio.google.com/ → Get API key
```

### 3. Run evaluation (local tracking so it shows in UI)

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
python test.py
```

You should see:

- Console: aggregated metrics and a preview of the evaluation table
- New file: `eval.csv` in the project root
- New/updated run in `./mlruns`

### 4. Start the MLflow UI (if not already running)

```bash
./run_mlflow_ui.sh
# Or: mlflow ui --backend-store-uri ./mlruns --port 5001
```

### 5. Open the UI and verify

- Open **http://localhost:5001** in your browser
- Go to **Experiments** → **"LLM Evaluation"**
- Open the latest run and check:
  - **Metrics**: e.g. token counts, latency, toxicity (if computed)
  - **Artifacts**: logged model
- Go to **Models** → **gemini-qa** and confirm the new version

### 6. Optional: test without API key (smoke test)

To only confirm the pipeline runs (it will fail during evaluation):

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
python test.py
# Expect: experiment + model registered, then ValueError about GEMINI_API_KEY
```

Then open http://localhost:5001 and confirm the experiment and **gemini-qa** model exist.

---

## How the Workflow Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│  test.py                                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Load config                                                         │
│     • .env or env: GEMINI_API_KEY, optional MLFLOW_TRACKING_URI          │
│     • If MLFLOW_TRACKING_URI set → use it (e.g. file:./mlruns)          │
│     • Else → DagsHub (mangalam-123/MLfLow)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  2. Define eval data                                                    │
│     • inputs: "What is MLflow?", "What is Spark?"                       │
│     • ground_truth: reference answers for each                           │
├─────────────────────────────────────────────────────────────────────────┤
│  3. Set experiment & start run                                          │
│     • mlflow.set_experiment("LLM Evaluation")                            │
│     • with mlflow.start_run():                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  4. Log Gemini as MLflow model                                          │
│     • GeminiQAModel (pyfunc): calls Gemini API per question              │
│     • mlflow.pyfunc.log_model(..., registered_model_name="gemini-qa")   │
│     • Model is registered in MLflow (local or DagsHub)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  5. Run evaluation                                                       │
│     • mlflow.evaluate(model_uri, eval_data, model_type="question-answering") │
│     • For each row: model predicts → metrics (token count, latency, …)  │
│     • Aggregated metrics logged to the run                               │
├─────────────────────────────────────────────────────────────────────────┤
│  6. Save outputs                                                         │
│     • results.tables["eval_results_table"] → eval.csv                   │
│     • Console: print metrics and table preview                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Where to look                                                          │
│  • Local UI:  http://localhost:5001  (when MLFLOW_TRACKING_URI=file:./mlruns) │
│  • DagsHub:   repo MLfLow → MLflow tab (when not using local URI)       │
│  • File:      eval.csv in project root                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### In short

| Step | What happens |
|------|----------------|
| 1 | Choose tracking: local (`file:./mlruns`) or DagsHub. |
| 2 | Eval dataset: 2 QA pairs (MLflow, Spark) with ground truth. |
| 3 | One MLflow run under experiment **"LLM Evaluation"**. |
| 4 | Gemini is wrapped as a pyfunc and logged as model **gemini-qa**. |
| 5 | MLflow runs **evaluate()**: calls Gemini per question, computes QA + latency/toxicity, logs metrics. |
| 6 | Results written to **eval.csv** and to the run in the UI. |

### Data flow

- **Input**: `eval_data` (inputs + ground_truth).
- **Model**: Gemini (via `GeminiQAModel.predict()` → Gemini API).
- **Output**: Run metrics in MLflow, per-row results in **eval.csv**, and the **gemini-qa** model in the Model Registry (visible in the UI).
