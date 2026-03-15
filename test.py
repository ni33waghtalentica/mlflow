import mlflow
import os
import sys
import time
import pandas as pd
import dagshub

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# MLflow question-answering evaluation requires tiktoken for token_count metric
try:
    import tiktoken  # noqa: F401
except ImportError:
    print(
        "Error: 'tiktoken' is required for MLflow QA evaluation (token_count metric).\n"
        "Install it with: pip install tiktoken",
        file=sys.stderr,
    )
    sys.exit(1)

from google import genai
from google.genai import errors as genai_errors

# Use your Gemini API key from environment or .env (GOOGLE_API_KEY or GEMINI_API_KEY)
def _get_gemini_client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        # Try .env in project root (set by main script so evaluation subprocess can find it)
        root = os.environ.get("MLFLOW_PROJECT_ROOT")
        if root:
            load_dotenv(os.path.join(root, ".env"))
    except ImportError:
        pass
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or in a .env file."
        )
    return genai.Client(api_key=api_key)

# Gemini model (use a supported model: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-flash-lite)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
# Seconds to wait between API calls to avoid rate limit (429)
GEMINI_REQUEST_DELAY = float(os.environ.get("GEMINI_REQUEST_DELAY", "2.0"))
# Max retries on 429 (quota exceeded), with wait
GEMINI_RETRY_MAX = int(os.environ.get("GEMINI_RETRY_MAX", "5"))


def _parse_429_retry_seconds(e: Exception) -> int:
    """Parse retry delay from 429 error (RetryInfo or message). Return seconds to wait."""
    import re
    # From error details: RetryInfo.retryDelay e.g. "44s"
    details = getattr(e, "details", None)
    if isinstance(details, dict):
        err = details.get("error", details)
        if isinstance(err, dict):
            for d in err.get("details", []) or []:
                if isinstance(d, dict) and "RetryInfo" in str(d.get("@type", "")):
                    delay_str = d.get("retryDelay", "")
                    m = re.match(r"(\d+)s", str(delay_str).strip())
                    if m:
                        return int(m.group(1)) + 1
    # Fallback: "Please retry in 44.281395397s"
    err_str = str(e).lower()
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", err_str)
    if m:
        return max(45, int(float(m.group(1))) + 1)
    return 60


def _generate_with_retry(client, model_name: str, contents: str):
    """Call generate_content with retry on 429 (quota/rate limit)."""
    last_err = None
    for attempt in range(GEMINI_RETRY_MAX):
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            return getattr(response, "text", None) or ""
        except genai_errors.ClientError as e:
            last_err = e
            code = getattr(e, "code", None)
            if code == 429 and attempt < GEMINI_RETRY_MAX - 1:
                wait = _parse_429_retry_seconds(e)
                time.sleep(wait)
                continue
            raise
    if last_err is not None:
        raise last_err


class GeminiQAModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for Gemini question-answering."""

    def __init__(self, system_prompt: str, model_name: str):
        self.system_prompt = system_prompt
        self.model_name = model_name

    def load_context(self, context):
        pass

    def predict(self, context, model_input, params=None):
        if hasattr(model_input, "columns") and "inputs" in model_input.columns:
            questions = model_input["inputs"].astype(str).tolist()
        else:
            questions = pd.DataFrame(model_input).iloc[:, 0].astype(str).tolist()
        client = _get_gemini_client()
        answers = []
        for i, q in enumerate(questions):
            if i > 0:
                time.sleep(GEMINI_REQUEST_DELAY)
            full_prompt = f"{self.system_prompt}\n\nQuestion: {q}"
            text = _generate_with_retry(client, self.model_name, full_prompt)
            answers.append(text if text else "")
        return pd.Series(answers)


# So the loaded model can find .env when evaluation runs (possibly different cwd)
os.environ.setdefault("MLFLOW_PROJECT_ROOT", os.path.abspath(os.path.dirname(__file__)))

# Use local tracking (e.g. ./mlruns) when MLFLOW_TRACKING_URI is set for local MLflow UI; else DagsHub
if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
else:
    dagshub.init(repo_owner='mangalam-123', repo_name='MLfLow', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/mangalam-123/mlflow.mlflow")

eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)
mlflow.set_experiment("LLM Evaluation")
with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    # Wrap Gemini as an MLflow pyfunc model.
    gemini_model = GeminiQAModel(system_prompt=system_prompt, model_name=GEMINI_MODEL)
    logged_model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=gemini_model,
        registered_model_name="gemini-qa",
    )

    # Use predefined question-answering metrics. Latency only for extra (toxicity/answer_similarity need evaluate/openai).
    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[mlflow.metrics.latency(),mlflow.metrics.toxicity()],
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # Evaluation result for each data record is available in `results.tables`.
    eval_table = results.tables["eval_results_table"]
    df=pd.DataFrame(eval_table)
    df.to_csv('eval.csv')
    print(f"See evaluation table below: \n{eval_table}")

