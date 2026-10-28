from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost

import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -----------------------------
# USE A DATABASE-BACKED MLflow TRACKING STORE
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MLFLOW_DIR = BASE_DIR / ".mlflow"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DB = MLFLOW_DIR / "tracking.db"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
experiment_name = "breast_cancer_classification"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(
        experiment_name,
        artifact_location=MLFLOW_ARTIFACTS.as_uri(),
    )
mlflow.set_experiment(experiment_name)

# -----------------------------
# LOAD DATA
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=5000),
    "xgboost": XGBClassifier(eval_metric="logloss", random_state=42)
}

best_model = None
best_score = 0
best_name = ""

# -----------------------------
# TRAIN + LOG
# -----------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log params + metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Use each framework's native MLflow flavor for model serialization.
        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

        print(f"{name} -> Accuracy: {acc}")

        # Track best model
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

# -----------------------------
# SAVE BEST MODEL LOCALLY
# -----------------------------
joblib.dump(best_model, "model.pkl")

print("\n======================")
print("BEST MODEL:", best_name)
print("BEST ACC:", best_score)
print("Saved: model.pkl")
print("======================")