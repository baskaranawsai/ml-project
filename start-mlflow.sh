#!/usr/bin/env bash
# Starts a local MLflow tracking server using sqlite backend and local artifact store (./mlruns).
# Usage: ./start-mlflow.sh

set -e

DB_FILE="mlflow.db"
ARTIFACT_ROOT="$(pwd)/mlruns"
PORT=5000

echo "Starting MLflow Tracking Server"
echo "DB: ${DB_FILE}"
echo "Artifacts: ${ARTIFACT_ROOT}"
echo "UI: http://127.0.0.1:${PORT}"

mlflow db upgrade "sqlite:///${DB_FILE}" || true

mlflow server \
  --backend-store-uri "sqlite:///${DB_FILE}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port ${PORT}
