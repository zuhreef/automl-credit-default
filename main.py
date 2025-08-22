# main.py
import os, json
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import train_test_split_data, build_preprocessor
from src.feature_selection import make_selector
from src.hyperparameter_tuning import tune_random_forest

DATA_PATH = "data/credit_clean.csv"
RESULTS_DIR = "results"

def main(n_trials: int = 30):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----- Load data
    print(f"[INFO] Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # ----- Split
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # ----- Build pipeline pieces
    pre = build_preprocessor(X_train)
    sel = make_selector()

    # Pre-transform once for faster CV/tuning
    base_pipe = Pipeline([("pre", pre), ("sel", sel)])
    Xtr = base_pipe.fit_transform(X_train, y_train)
    Xte = base_pipe.transform(X_test)

    # ----- Quick baseline CV with a vanilla RF
    rf0 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1cv = cross_val_score(rf0, Xtr, y_train, cv=cv,
                           scoring=make_scorer(f1_score),
                           n_jobs=-1).mean()
    print(f"[INFO] Baseline RF CV F1: {f1cv:.4f}")

    # ----- Optuna tune RF on transformed Xtr
    print(f"[INFO] Tuning RandomForest for {n_trials} trials…")
    study = tune_random_forest(Xtr, y_train.values, n_trials=n_trials)
    print("[INFO] Best params:", study.best_params)

    # ----- Train final full pipeline and evaluate
    best_rf = RandomForestClassifier(**study.best_params)
    final_pipe = Pipeline([("pre", pre), ("sel", sel), ("clf", best_rf)])
    final_pipe.fit(X_train, y_train)

    y_pred  = final_pipe.predict(X_test)
    y_proba = final_pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "baseline_rf_cv_f1": float(f1cv),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_test": int(len(y_test)),
    }
    print("[INFO] Test metrics:", metrics)

    # ----- Save artifacts
    dump(final_pipe, os.path.join(RESULTS_DIR, "final_pipeline.joblib"))
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Saved model to results/final_pipeline.joblib")
    print("[INFO] Saved metrics to results/metrics.json")

if __name__ == "__main__":
    # change n_trials here if you want
    main(n_trials=30)
