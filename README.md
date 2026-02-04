# ml-project

This repository demonstrates an end-to-end machine learning project with MLOps practices using MLflow.

It includes:
- A reproducible training script that trains a scikit-learn model and logs params, metrics and artifacts to MLflow.
- A local MLflow tracking server script for development.
- CI workflow that runs the training script to validate the pipeline.

## Prerequisites

- Python 3.9+ or use the provided `environment.yml`.
- Docker (optional, to run MLflow UI in a container).
- MLflow (installed via `pip install -r requirements.txt`).

## Quickstart (local)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use conda:

```bash
conda env create -f environment.yml
conda activate ml-project
```

2. Start a local MLflow tracking server (stores metadata in `mlflow.db` and artifacts in `./mlruns`):

```bash
# Make executable: chmod +x start-mlflow.sh
./start-mlflow.sh
# MLflow UI will be available at http://127.0.0.1:5000
```

3. Run training (logs go to the MLflow server):

```bash
# Example run
python -m src.train --n_estimators 50 --max_depth 5
```

4. View experiments:
Open http://127.0.0.1:5000

## Files

- `src/train.py` — training script using scikit-learn; logs to MLflow.
- `start-mlflow.sh` — helper to start MLflow server with sqlite backend.
- `.github/workflows/ci.yml` — CI job that installs deps and runs the training script.
- `requirements.txt`, `environment.yml` — reproducible environment descriptors.

## Usage notes

- By default the training script uses environment variable `MLFLOW_TRACKING_URI` if set; otherwise it will call `mlflow.set_tracking_uri("http://127.0.0.1:5000")` so you can point it to a remote tracking server easily.
- For production/remote MLflow servers, configure `MLFLOW_TRACKING_URI` and artifact storage (S3, Azure Blob, GCS, etc.) as needed.

## CI / MLOps ideas

- Use the GitHub Actions workflow to run training, publish metrics to MLflow, and create model registry entries (requires a remote MLflow server with authentication).
- Add model validation and integration tests in CI before promoting models.

## Next steps

- If you want, I can push these files into a branch and open a PR (recommended) or create a release artifact, or extend the pipeline to register models in MLflow Model Registry.