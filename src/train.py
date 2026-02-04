"""
Train a simple model and log parameters, metrics and artifacts to MLflow.

Usage examples:
    python -m src.train --n_estimators 50 --max_depth 5
"""
import os
import argparse
import tempfile
import json

import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple sklearn model and log to MLflow")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in RandomForest")
    parser.add_argument("--max_depth", type=int, default=None, help="Max depth for RandomForest")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default="ml-project")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    return args


def prepare_data(random_state=42, test_size=0.2):
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def main():
    args = parse_args()

    # allow overriding tracking URI via env var
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "test_size": args.test_size,
            "random_state": args.random_state,
        })

        X_train, X_test, y_train, y_test = prepare_data(random_state=args.random_state, test_size=args.test_size)

        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", float(acc))

        # Save model as artifact
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.joblib")
            joblib.dump(clf, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        # Example: save a small validation report
        report = {
            "accuracy": float(acc),
            "n_test": int(len(y_test)),
            "run_id": run.info.run_id
        }
        with open("validation_report.json", "w") as f:
            json.dump(report, f)
        mlflow.log_artifact("validation_report.json", artifact_path="reports")

        # Log model via MLflow sklearn model format (optional)
        mlflow.sklearn.log_model(clf, artifact_path="sklearn-model")

        print(f"Run completed. Run ID: {run.info.run_id}. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
