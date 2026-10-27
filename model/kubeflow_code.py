from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


# ----------------------------------------------------------------------
# 1. Data ingestion
# ----------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn"],
)
def ingestion_op(output_data: Output[Dataset]) -> None:
    import pandas as pd
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    if "target" not in df.columns:
        raise ValueError("Breast cancer dataset missing target column")
    print(f"Loaded breast cancer dataset with shape: {df.shape}")
    df.to_csv(output_data.path, index=False)


# ----------------------------------------------------------------------
# 2. Validation
# ----------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "numpy"])
def validation_op(data: Input[Dataset]) -> str:
    import pandas as pd
    target_col = "target"
    df = pd.read_csv(data.path, low_memory=False)
    if df.empty:
        raise ValueError("Dataset is empty")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    counts = df[target_col].value_counts(dropna=False)
    if len(counts) < 2:
        raise ValueError("Target must contain both classes")
    if counts.min() < 2:
        raise ValueError("Each class must have at least 2 rows")
    return f"Validation passed: rows={len(df)}, cols={len(df.columns)}"


# ----------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "numpy"])
def feature_op(data: Input[Dataset], output_data: Output[Dataset]) -> None:
    import numpy as np
    import pandas as pd
    df = pd.read_csv(data.path, low_memory=False)
    numeric_cols = [c for c in df.columns if c != "target" and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 3:
        df["mean_first_three"] = df[numeric_cols[:3]].mean(axis=1)
        df["std_first_three"] = df[numeric_cols[:3]].std(axis=1)
        df["sum_first_three"] = df[numeric_cols[:3]].sum(axis=1)
    if "mean radius" in df.columns and "mean perimeter" in df.columns:
        denom = df["mean perimeter"].replace(0, np.nan)
        df["radius_perimeter_ratio"] = df["mean radius"] / denom
        df["radius_perimeter_ratio"] = df["radius_perimeter_ratio"].replace([np.inf, -np.inf], np.nan)
    if "mean area" in df.columns and "mean radius" in df.columns:
        denom = df["mean radius"].replace(0, np.nan)
        df["area_radius_ratio"] = df["mean area"] / denom
        df["area_radius_ratio"] = df["area_radius_ratio"].replace([np.inf, -np.inf], np.nan)
    print(f"Feature engineering complete. Shape: {df.shape}")
    df.to_csv(output_data.path, index=False)


# ----------------------------------------------------------------------
# 4. Split into train, calibration, test
# ----------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "scikit-learn"])
def split_calibration_op(
    data: Input[Dataset],
    train_data: Output[Dataset],
    cal_data: Output[Dataset],
    test_data: Output[Dataset],
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    random_state: int = 42,
) -> None:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    target_col = "target"
    df = pd.read_csv(data.path, low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_frac), random_state=random_state, stratify=df[target_col]
    )
    test_frac = 1 - train_frac - cal_frac
    cal_df, test_df = train_test_split(
        temp_df, test_size=test_frac / (cal_frac + test_frac), random_state=random_state,
        stratify=temp_df[target_col]
    )
    print(f"Split complete. Train shape: {train_df.shape}, Cal shape: {cal_df.shape}, Test shape: {test_df.shape}")
    train_df.to_csv(train_data.path, index=False)
    cal_df.to_csv(cal_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)


# ----------------------------------------------------------------------
# 5. Preprocessing (fit on train, apply to cal and test)
# ----------------------------------------------------------------------
@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "numpy"])
def preprocess_op(
    train_data: Input[Dataset],
    cal_data: Input[Dataset],
    test_data: Input[Dataset],
    processed_train_data: Output[Dataset],
    processed_cal_data: Output[Dataset],
    processed_test_data: Output[Dataset],
    high_cardinality_threshold: int = 50,
) -> None:
    import numpy as np
    import pandas as pd

    def preprocess_single(df, target_col, fit=False, median_vals=None, cat_mappings=None):
        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]

        if fit:
            median_vals = {}
            for col in num_cols:
                median_vals[col] = X[col].median()
            cat_mappings = {}
            for col in cat_cols:
                X[col] = X[col].astype("object").fillna("missing")
                nunique = X[col].nunique(dropna=False)
                if nunique > high_cardinality_threshold:
                    freq = X[col].value_counts(normalize=True)
                    cat_mappings[col] = ("freq", freq)
                else:
                    categories = list(X[col].astype(str).value_counts().index)
                    mapping = {v: i+1 for i, v in enumerate(categories)}
                    cat_mappings[col] = ("code", mapping)
        # Apply
        X_out = pd.DataFrame()
        for col in num_cols:
            median = median_vals.get(col, X[col].median())
            X_out[col] = X[col].fillna(median)
            missing = X[col].isna().astype(int)
            X_out[f"{col}_missing"] = missing
        for col in cat_cols:
            X[col] = X[col].astype("object").fillna("missing")
            mtype, mapping = cat_mappings[col]
            if mtype == "freq":
                X_out[f"{col}_freq"] = X[col].map(mapping).fillna(0.0)
            else:
                X_out[f"{col}_code"] = X[col].astype(str).map(mapping).fillna(0).astype(int)
        return X_out, y, median_vals, cat_mappings

    target_col = "target"
    train_df = pd.read_csv(train_data.path, low_memory=False)
    cal_df = pd.read_csv(cal_data.path, low_memory=False)
    test_df = pd.read_csv(test_data.path, low_memory=False)

    X_train, y_train, median_vals, cat_mappings = preprocess_single(train_df, target_col, fit=True)
    X_cal, y_cal, _, _ = preprocess_single(cal_df, target_col, fit=False, median_vals=median_vals, cat_mappings=cat_mappings)
    X_test, y_test, _, _ = preprocess_single(test_df, target_col, fit=False, median_vals=median_vals, cat_mappings=cat_mappings)

    train_out = pd.concat([X_train, y_train], axis=1)
    cal_out = pd.concat([X_cal, y_cal], axis=1)
    test_out = pd.concat([X_test, y_test], axis=1)

    train_out = train_out.replace([np.inf, -np.inf], np.nan).fillna(0)
    cal_out = cal_out.replace([np.inf, -np.inf], np.nan).fillna(0)
    test_out = test_out.replace([np.inf, -np.inf], np.nan).fillna(0)

    train_out.to_csv(processed_train_data.path, index=False)
    cal_out.to_csv(processed_cal_data.path, index=False)
    test_out.to_csv(processed_test_data.path, index=False)


# ----------------------------------------------------------------------
# 6. Training components (train_1 to train_5)
# ----------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "joblib"],
)
def train_1(data: Input[Dataset], model: Output[Model]) -> None:
    """Train XGBoost (standard)"""
    import joblib
    import pandas as pd
    from xgboost import XGBClassifier

    target_col = "target"
    df = pd.read_csv(data.path, low_memory=False)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    clf = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)
    joblib.dump(clf, model.path)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "joblib"],
)
def train_2(data: Input[Dataset], model: Output[Model]) -> None:
    """Train XGBoost cost-sensitive"""
    import joblib
    import pandas as pd
    from xgboost import XGBClassifier

    target_col = "target"
    df = pd.read_csv(data.path, low_memory=False)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = neg / max(pos, 1)
    clf = XGBClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)
    joblib.dump(clf, model.path)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "imbalanced-learn", "joblib"],
)
def train_3(data: Input[Dataset], model: Output[Model]) -> None:
    """Train XGBoost with SMOTE oversampling"""
    import joblib
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier

    target_col = "target"
    df = pd.read_csv(data.path, low_memory=False)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    minority_count = int(y.value_counts().min())
    if minority_count >= 2:
        k_neighbors = min(5, minority_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    clf = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    clf.fit(X_res, y_res)
    joblib.dump(clf, model.path)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "lightgbm", "joblib"],
)
def train_4(data: Input[Dataset], model: Output[Model]) -> None:
    """Train LightGBM cost-sensitive"""
    import joblib
    import pandas as pd
    from lightgbm import LGBMClassifier

    target_col = "target"
    df = pd.read_csv(data.path, low_memory=False)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = neg / max(pos, 1)
    clf = LGBMClassifier(
        n_estimators=200, max_depth=-1, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)
    joblib.dump(clf, model.path)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
)
def train_5(data: Input[Dataset], model: Output[Model]) -> None:
    """Train hybrid Random Forest with feature selection"""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.pipeline import Pipeline

    target_col = "target"
    df = pd.read_csv(data.path, low_memory=False)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    selector_model = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)
    final_rf = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1)
    hybrid = Pipeline([
        ("feature_selection", SelectFromModel(selector_model, threshold="median")),
        ("rf", final_rf),
    ])
    hybrid.fit(X, y)
    joblib.dump(hybrid, model.path)


# ----------------------------------------------------------------------
# 7. Evaluation for each model (standard metrics)
# ----------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib", "xgboost", "lightgbm"],
)
def evaluate_op(
    data: Input[Dataset],
    model: Input[Model],
    fraud_loss_cost: float = 500.0,
    false_alarm_cost: float = 10.0,
) -> NamedTuple(
    "Metrics",
    [
        ("accuracy", float),
        ("precision", float),
        ("recall", float),
        ("f1", float),
        ("auc_roc", float),
        ("false_positives", float),
        ("false_negatives", float),
        ("business_cost", float),
    ],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    target_col = "target"
    df = pd.read_csv(data.path, low_memory=False)
    y_true = df[target_col]
    X = df.drop(columns=[target_col])
    clf = joblib.load(model.path)
    preds = clf.predict(X)
    try:
        scores = clf.predict_proba(X)[:, 1]
    except Exception:
        try:
            scores = clf.decision_function(X)
        except Exception:
            scores = preds
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc_roc = roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else 0.5
    business_cost = float(fn * fraud_loss_cost + fp * false_alarm_cost)
    return (float(accuracy), float(precision), float(recall), float(f1),
            float(auc_roc), float(fp), float(fn), float(business_cost))


# ----------------------------------------------------------------------
# 8. Model selection (choose best from 5 models)
# ----------------------------------------------------------------------
@dsl.component(base_image="python:3.11")
def select_best_model_op(
    model_1: Input[Model], recall_1: float, auc_1: float, cost_1: float,
    model_2: Input[Model], recall_2: float, auc_2: float, cost_2: float,
    model_3: Input[Model], recall_3: float, auc_3: float, cost_3: float,
    model_4: Input[Model], recall_4: float, auc_4: float, cost_4: float,
    model_5: Input[Model], recall_5: float, auc_5: float, cost_5: float,
    selected_model: Output[Model],
) -> NamedTuple("Selection", [("best_model_name", str), ("best_recall", float), ("best_auc", float), ("best_business_cost", float)]):
    import shutil
    candidates = [
        {"name": "train_1 (XGBoost standard)", "path": model_1.path, "recall": recall_1, "auc": auc_1, "cost": cost_1},
        {"name": "train_2 (XGBoost cost-sensitive)", "path": model_2.path, "recall": recall_2, "auc": auc_2, "cost": cost_2},
        {"name": "train_3 (XGBoost SMOTE)", "path": model_3.path, "recall": recall_3, "auc": auc_3, "cost": cost_3},
        {"name": "train_4 (LightGBM cost-sensitive)", "path": model_4.path, "recall": recall_4, "auc": auc_4, "cost": cost_4},
        {"name": "train_5 (Hybrid RF + FS)", "path": model_5.path, "recall": recall_5, "auc": auc_5, "cost": cost_5},
    ]
    best = sorted(candidates, key=lambda x: (-x["recall"], x["cost"], -x["auc"]))[0]
    shutil.copy2(best["path"], selected_model.path)
    print(f"Best model selected: {best['name']} | recall={best['recall']:.4f} | auc={best['auc']:.4f} | cost={best['cost']:.2f}")
    return (best["name"], float(best["recall"]), float(best["auc"]), float(best["cost"]))


# ----------------------------------------------------------------------
# 9. NOVEL COMPONENT: Conformal Prediction Calibration
# ----------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib", "xgboost", "lightgbm"],
)
def conformal_calibration_op(
    model: Input[Model],
    cal_data: Input[Dataset],
    calibrated_model: Output[Model],
    alpha: float = 0.05,
) -> NamedTuple("CalibMetrics", [("threshold", float), ("empirical_coverage_cal", float)]):
    import joblib
    import numpy as np
    import pandas as pd

    df = pd.read_csv(cal_data.path)
    target_col = "target"
    y_cal = df[target_col]
    X_cal = df.drop(columns=[target_col])

    clf = joblib.load(model.path)
    proba = clf.predict_proba(X_cal)
    true_probs = proba[np.arange(len(y_cal)), y_cal]
    nonconformity = 1 - true_probs

    n = len(nonconformity)
    quantile = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(nonconformity, quantile, method="higher")

    pred_sets = []
    for i in range(len(X_cal)):
        probs = clf.predict_proba(X_cal.iloc[[i]])[0]
        ncf = 1 - probs
        included = ncf <= threshold
        pred_set = list(np.where(included)[0])
        pred_sets.append(pred_set)
    coverages = [1 if y_cal.iloc[i] in pred_set else 0 for i, pred_set in enumerate(pred_sets)]
    empirical_cov = np.mean(coverages)

    calibrated_dict = {
        "model": clf,
        "threshold": float(threshold),
        "alpha": alpha,
        "classes": clf.classes_.tolist()
    }
    joblib.dump(calibrated_dict, calibrated_model.path)

    print(f"Conformal calibration: alpha={alpha}, threshold={threshold:.4f}, empirical coverage = {empirical_cov:.3f}")
    return (float(threshold), float(empirical_cov))


# ----------------------------------------------------------------------
# 10. Evaluation with conformal prediction (includes all needed libs)
# ----------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib", "xgboost", "lightgbm"],
)
def evaluate_conformal_op(
    calibrated_model: Input[Model],
    test_data: Input[Dataset],
) -> NamedTuple("TestMetrics", [("coverage", float), ("avg_set_size", float), ("accuracy_original", float)]):
    import numpy as np
    import pandas as pd
    import joblib

    df = pd.read_csv(test_data.path)
    target_col = "target"
    y_test = df[target_col]
    X_test = df.drop(columns=[target_col])

    calib = joblib.load(calibrated_model.path)
    clf = calib["model"]
    threshold = calib["threshold"]

    proba = clf.predict_proba(X_test)
    pred_sets = []
    for i in range(len(X_test)):
        probs = proba[i]
        ncf = 1 - probs
        included = ncf <= threshold
        pred_set = list(np.where(included)[0])
        pred_sets.append(pred_set)

    coverage = np.mean([1 if y_test.iloc[i] in pred_set else 0 for i, pred_set in enumerate(pred_sets)])
    avg_set_size = np.mean([len(ps) for ps in pred_sets])
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Test results: Coverage = {coverage:.3f} (target = {1-calib['alpha']:.3f}), Avg set size = {avg_set_size:.2f}, Accuracy = {accuracy:.3f}")
    return (float(coverage), float(avg_set_size), float(accuracy))


# ----------------------------------------------------------------------
# 11. Pre-deploy checks (install ALL model libraries: sklearn, xgboost, lightgbm)
# ----------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["joblib", "numpy", "psutil", "scikit-learn", "xgboost", "lightgbm"]
)
def pre_deploy_checks_op(
    model: Input[Model],
    max_size_mb: float = 50.0,
) -> str:
    import os
    import time
    import numpy as np
    import joblib
    import psutil

    size_bytes = os.path.getsize(model.path)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_size_mb:
        raise RuntimeError(f"Model too large: {size_mb:.2f} MB > {max_size_mb} MB")

    calib = joblib.load(model.path)
    clf = calib["model"]
    n_features = getattr(clf, "n_features_in_", 30)
    dummy = np.random.randn(1, n_features)
    start = time.perf_counter()
    _ = clf.predict(dummy)
    latency_ms = (time.perf_counter() - start) * 1000

    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Pre-deploy checks: size={size_mb:.2f} MB, latency={latency_ms:.2f} ms, memory={mem_mb:.1f} MB")
    return f"Checks passed: size {size_mb:.2f} MB, latency {latency_ms:.2f} ms"


# ----------------------------------------------------------------------
# 12. Deployment stub
# ----------------------------------------------------------------------
@dsl.component(base_image="python:3.11")
def deploy_op(
    model: Input[Model],
    coverage: float,
    avg_set_size: float,
) -> str:
    print(f"Deploying conformalized model with coverage={coverage:.3f}, average set size={avg_set_size:.2f}")
    print(f"Model artifact path: {model.path}")
    return "Deployment triggered"


# ----------------------------------------------------------------------
# 13. Main pipeline definition
# ----------------------------------------------------------------------
@dsl.pipeline(
    name="novel-conformal-ml-pipeline",
    description="End-to-end MLOps pipeline with 5 models, model selection, and conformal prediction.",
)
def conformal_prediction_pipeline(
    recall_threshold: float = 0.80,
    auc_threshold: float = 0.85,
    alpha: float = 0.05,
    coverage_threshold: float = 0.90,
):
    # Data preparation
    ingest = ingestion_op().set_retry(2).set_cpu_limit("1").set_memory_limit("1Gi")
    validate = validation_op(data=ingest.outputs["output_data"]).set_cpu_limit("1").set_memory_limit("1Gi")
    feature = feature_op(data=ingest.outputs["output_data"]).after(validate).set_cpu_limit("1").set_memory_limit("1Gi")
    split = split_calibration_op(
        data=feature.outputs["output_data"],
        train_frac=0.6,
        cal_frac=0.2,
    ).set_retry(2).set_cpu_limit("1").set_memory_limit("1Gi")
    preproc = preprocess_op(
        train_data=split.outputs["train_data"],
        cal_data=split.outputs["cal_data"],
        test_data=split.outputs["test_data"],
    ).set_retry(2).set_cpu_limit("1").set_memory_limit("2Gi")

    # Training (5 models)
    train1 = train_1(data=preproc.outputs["processed_train_data"]).set_retry(2).set_cpu_limit("1").set_memory_limit("2Gi")
    train2 = train_2(data=preproc.outputs["processed_train_data"]).set_retry(2).set_cpu_limit("1").set_memory_limit("2Gi")
    train3 = train_3(data=preproc.outputs["processed_train_data"]).set_retry(2).set_cpu_limit("1").set_memory_limit("2Gi")
    train4 = train_4(data=preproc.outputs["processed_train_data"]).set_retry(2).set_cpu_limit("1").set_memory_limit("2Gi")
    train5 = train_5(data=preproc.outputs["processed_train_data"]).set_retry(2).set_cpu_limit("1").set_memory_limit("2Gi")

    # Evaluation (5 models)
    eval1 = evaluate_op(data=preproc.outputs["processed_test_data"], model=train1.outputs["model"])
    eval2 = evaluate_op(data=preproc.outputs["processed_test_data"], model=train2.outputs["model"])
    eval3 = evaluate_op(data=preproc.outputs["processed_test_data"], model=train3.outputs["model"])
    eval4 = evaluate_op(data=preproc.outputs["processed_test_data"], model=train4.outputs["model"])
    eval5 = evaluate_op(data=preproc.outputs["processed_test_data"], model=train5.outputs["model"])

    # Model selection
    select = select_best_model_op(
        model_1=train1.outputs["model"], recall_1=eval1.outputs["recall"], auc_1=eval1.outputs["auc_roc"], cost_1=eval1.outputs["business_cost"],
        model_2=train2.outputs["model"], recall_2=eval2.outputs["recall"], auc_2=eval2.outputs["auc_roc"], cost_2=eval2.outputs["business_cost"],
        model_3=train3.outputs["model"], recall_3=eval3.outputs["recall"], auc_3=eval3.outputs["auc_roc"], cost_3=eval3.outputs["business_cost"],
        model_4=train4.outputs["model"], recall_4=eval4.outputs["recall"], auc_4=eval4.outputs["auc_roc"], cost_4=eval4.outputs["business_cost"],
        model_5=train5.outputs["model"], recall_5=eval5.outputs["recall"], auc_5=eval5.outputs["auc_roc"], cost_5=eval5.outputs["business_cost"],
    ).set_retry(2).set_cpu_limit("1").set_memory_limit("1Gi")

    # Conformal calibration
    calibrate = conformal_calibration_op(
        model=select.outputs["selected_model"],
        cal_data=preproc.outputs["processed_cal_data"],
        alpha=alpha,
    ).set_cpu_limit("1").set_memory_limit("2Gi")

    # Conformal evaluation on test set
    evaluate = evaluate_conformal_op(
        calibrated_model=calibrate.outputs["calibrated_model"],
        test_data=preproc.outputs["processed_test_data"],
    ).set_cpu_limit("1").set_memory_limit("1Gi")

    # ----- Combined condition -----
    deploy_condition = (
        (select.outputs["best_recall"] >= recall_threshold) and
        (select.outputs["best_auc"] >= auc_threshold) and
        (evaluate.outputs["coverage"] >= coverage_threshold)
    )

    with dsl.If(deploy_condition):
        predeploy = pre_deploy_checks_op(
            model=calibrate.outputs["calibrated_model"],
        ).set_cpu_limit("1").set_memory_limit("1Gi")
        deploy = deploy_op(
            model=calibrate.outputs["calibrated_model"],
            coverage=evaluate.outputs["coverage"],
            avg_set_size=evaluate.outputs["avg_set_size"],
        ).after(predeploy)
        deploy.set_cpu_limit("1").set_memory_limit("1Gi")
    # -------------------------------------------------


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=conformal_prediction_pipeline,
        package_path="novel_breast_cancer_pipeline.yaml"
    )