#!/usr/bin/env bash
# Run MLflow UI to view experiments. Use local store (./mlruns) so runs from
# test.py (with MLFLOW_TRACKING_URI=file:./mlruns) appear here.
cd "$(dirname "$0")"
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5001
