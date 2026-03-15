"""
Add more MLflow experiments (industry-relevant) to the local UI.
Run with:  MLFLOW_TRACKING_URI=file:./mlruns  python run_extra_experiments.py
Uses same .env for Gemini; sklearn experiments need no API key.
"""
import os
import time
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import mlflow

# Use local tracking so runs appear in mlflow ui
os.environ.setdefault("MLFLOW_PROJECT_ROOT", os.path.abspath(os.path.dirname(__file__)))
if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
else:
    mlflow.set_tracking_uri("file:./mlruns")


def run_prompt_comparison():
    """Experiment: compare different system prompts for QA (prompt engineering)."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Skipping Prompt Comparison (no GEMINI_API_KEY in .env)")
        return
    try:
        from google import genai
        from google.genai import errors as genai_errors
    except ImportError:
        print("Skipping Prompt Comparison (google-genai not installed)")
        return

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    questions = ["What is MLflow?", "What is Spark?"]
    prompts = [
        ("one_sentence", "Answer in exactly one short sentence."),
        ("two_sentences", "Answer in two sentences."),
        ("detailed", "Answer in 2-3 sentences with key details."),
    ]

    mlflow.set_experiment("Prompt Comparison (QA)")
    for prompt_id, system_prompt in prompts:
        with mlflow.start_run(run_name=prompt_id):
            mlflow.log_param("prompt_style", prompt_id)
            mlflow.log_param("system_prompt", system_prompt)
            latencies = []
            token_counts = []
            for q in questions:
                t0 = time.perf_counter()
                try:
                    r = client.models.generate_content(
                        model=model_name,
                        contents=f"{system_prompt}\n\nQuestion: {q}",
                    )
                    text = getattr(r, "text", None) or ""
                except genai_errors.ClientError:
                    text = ""
                latencies.append(time.perf_counter() - t0)
                token_counts.append(max(1, len(text.split())))
                time.sleep(1.0)
            mlflow.log_metric("avg_latency_sec", np.mean(latencies))
            mlflow.log_metric("total_tokens_approx", sum(token_counts))
            mlflow.log_metric("num_questions", len(questions))
    print("Done: Prompt Comparison (QA)")


def run_regression_experiment():
    """Experiment: sklearn Ridge regression (e.g. diabetes prediction)."""
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    mlflow.set_experiment("Regression (Diabetes)")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        with mlflow.start_run(run_name=f"ridge_alpha_{alpha}"):
            mlflow.log_param("alpha", alpha)
            model = Ridge(alpha=alpha).fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, "model")
    print("Done: Regression (Diabetes)")


def run_classification_experiment():
    """Experiment: sklearn classification (digit recognition)."""
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    mlflow.set_experiment("Classification (Digits)")
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    for C in [0.1, 1.0, 10.0]:
        with mlflow.start_run(run_name=f"logreg_C_{C}"):
            mlflow.log_param("C", C)
            model = LogisticRegression(max_iter=1000, C=C, random_state=42).fit(
                X_train, y_train
            )
            pred = model.predict(X_test)
            mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
            mlflow.log_metric("f1_weighted", f1_score(y_test, pred, average="weighted"))
            mlflow.sklearn.log_model(model, "model")
    print("Done: Classification (Digits)")


def run_hyperparameter_sweep():
    """Experiment: simple hyperparameter sweep (Ridge alpha)."""
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score

    mlflow.set_experiment("Hyperparameter Sweep")
    X, y = make_regression(n_samples=500, n_features=10, noise=0.2, random_state=42)
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        with mlflow.start_run(run_name=f"alpha_{alpha}"):
            mlflow.log_param("alpha", alpha)
            model = Ridge(alpha=alpha)
            scores = -cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error")
            mlflow.log_metric("cv_mse_mean", float(scores.mean()))
            mlflow.log_metric("cv_mse_std", float(scores.std()))
    print("Done: Hyperparameter Sweep")


def _log_cancer_run_enhancements(
    X_tr, y_tr, X_te, y_te, pred, dataset_name, model_name, n_classes,
    class_names=None, register_model_name=None
):
    """Log datasets, artifacts (confusion matrix, sample CSV), tags, description, duration."""
    import tempfile
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Dataset logging (shows under "Datasets" in UI)
    try:
        train_df = pd.DataFrame(X_tr).assign(target=y_tr)
        dataset = mlflow.data.from_pandas(train_df, source=f"{dataset_name}_train.csv")
        mlflow.log_input(dataset, context="training")
    except Exception:
        pass

    # Tags for filtering and workflow
    mlflow.set_tag("project", "cancer_classification")
    mlflow.set_tag("data_source", dataset_name)
    mlflow.set_tag("model_type", model_name)
    mlflow.set_tag("n_classes", str(n_classes))

    # Run description (shows in Overview)
    mlflow.set_tag(
        "mlflow.note.content",
        f"Cancer classification run: {model_name} on {dataset_name}. "
        f"Classes={n_classes}. Logs datasets, confusion matrix, and sample data."
    )

    # Extra metrics
    from sklearn.metrics import precision_score, recall_score
    mlflow.log_metric("precision_weighted", precision_score(y_te, pred, average="weighted", zero_division=0))
    mlflow.log_metric("recall_weighted", recall_score(y_te, pred, average="weighted", zero_division=0))

    # Confusion matrix artifact
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_te, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names or list(range(n_classes)))
    disp.plot(ax=ax, values_format="d")
    plt.tight_layout()
    mlflow.log_figure(fig, "artifacts/confusion_matrix.png")
    plt.close()

    # Sample data CSV artifact (viewable in Artifacts tab)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        tmp_path = f.name
    sample = pd.DataFrame(X_te[:20]).assign(actual=list(y_te[:20]), predicted=list(pred[:20]))
    sample.to_csv(tmp_path, index=False)
    mlflow.log_artifact(tmp_path, artifact_path="artifacts")
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    # Log (and optionally register) model so Artifacts + Model Registry are populated
    if register_model_name:
        mlflow.sklearn.log_model(
            _log_cancer_run_enhancements._last_model,
            "model",
            registered_model_name=register_model_name,
        )
    else:
        mlflow.sklearn.log_model(_log_cancer_run_enhancements._last_model, "model")


def run_cancer_experiment():
    """Experiment: cancer classification — real (breast) + synthetic (multi-type)."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    mlflow.set_experiment("Cancer (Breast + Multi-type)")
    cancer_types = ["Breast", "Lung", "Prostate", "Colon", "Melanoma"]

    # --- Run 1: Real breast cancer data (malignant vs benign) ---
    X_real, y_real = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    with mlflow.start_run(run_name="breast_cancer_rf"):
        t0 = time.perf_counter()
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mlflow.log_metric("training_duration_sec", time.perf_counter() - t0)
        mlflow.log_param("dataset", "breast_cancer_wisconsin")
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_classes", 2)
        mlflow.log_metric("accuracy", accuracy_score(y_te, pred))
        mlflow.log_metric("f1_weighted", f1_score(y_te, pred, average="weighted"))
        _log_cancer_run_enhancements._last_model = model
        _log_cancer_run_enhancements(
            X_tr, y_tr, X_te, y_te, pred,
            "breast_cancer_wisconsin", "RandomForestClassifier", 2,
            class_names=["benign", "malignant"],
            register_model_name="cancer-classifier",
        )
    with mlflow.start_run(run_name="breast_cancer_logreg"):
        t0 = time.perf_counter()
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mlflow.log_metric("training_duration_sec", time.perf_counter() - t0)
        mlflow.log_param("dataset", "breast_cancer_wisconsin")
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("n_classes", 2)
        mlflow.log_metric("accuracy", accuracy_score(y_te, pred))
        mlflow.log_metric("f1_weighted", f1_score(y_te, pred, average="weighted"))
        _log_cancer_run_enhancements._last_model = model
        _log_cancer_run_enhancements(
            X_tr, y_tr, X_te, y_te, pred,
            "breast_cancer_wisconsin", "LogisticRegression", 2,
            class_names=["benign", "malignant"],
        )

    # --- Run 2: Synthetic multi-cancer-type data (dummy for demo) ---
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(42)
    n_samples = 800
    n_features = 12
    X_syn = rng.randn(n_samples, n_features).astype(np.float32)
    n_per_class = n_samples // 5
    for i in range(5):
        start, end = i * n_per_class, (i + 1) * n_per_class
        X_syn[start:end, i % n_features] += 2.0 * (i + 1)
    y_syn = np.repeat(np.arange(5), n_per_class)[:n_samples]
    scaler = StandardScaler()
    X_syn = scaler.fit_transform(X_syn)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn
    )
    with mlflow.start_run(run_name="multi_cancer_type_rf"):
        t0 = time.perf_counter()
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mlflow.log_metric("training_duration_sec", time.perf_counter() - t0)
        mlflow.log_param("dataset", "synthetic_multi_cancer")
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_classes", 5)
        mlflow.log_param("cancer_types", "Breast,Lung,Prostate,Colon,Melanoma")
        mlflow.log_metric("accuracy", accuracy_score(y_te, pred))
        mlflow.log_metric("f1_weighted", f1_score(y_te, pred, average="weighted", zero_division=0))
        _log_cancer_run_enhancements._last_model = model
        _log_cancer_run_enhancements(
            X_tr, y_tr, X_te, y_te, pred,
            "synthetic_multi_cancer", "RandomForestClassifier", 5,
            class_names=cancer_types,
        )
    with mlflow.start_run(run_name="multi_cancer_type_logreg"):
        t0 = time.perf_counter()
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mlflow.log_metric("training_duration_sec", time.perf_counter() - t0)
        mlflow.log_param("dataset", "synthetic_multi_cancer")
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("n_classes", 5)
        mlflow.log_param("cancer_types", "Breast,Lung,Prostate,Colon,Melanoma")
        mlflow.log_metric("accuracy", accuracy_score(y_te, pred))
        mlflow.log_metric("f1_weighted", f1_score(y_te, pred, average="weighted", zero_division=0))
        _log_cancer_run_enhancements._last_model = model
        _log_cancer_run_enhancements(
            X_tr, y_tr, X_te, y_te, pred,
            "synthetic_multi_cancer", "LogisticRegression", 5,
            class_names=cancer_types,
            register_model_name="cancer-classifier",
        )

    print("Done: Cancer (Breast + Multi-type)")


if __name__ == "__main__":
    print("Running extra experiments (tracking: file:./mlruns)...")
    run_regression_experiment()
    run_classification_experiment()
    run_hyperparameter_sweep()
    run_cancer_experiment()
    run_prompt_comparison()
    print("Refresh MLflow UI at http://localhost:5001 to see new experiments.")
