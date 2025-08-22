from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def candidate_models() -> Dict[str, object]:
    """
    Return a small set of strong baseline classifiers.
    """
    return {
        "logreg": LogisticRegression(max_iter=2000, n_jobs=None),  # liblinear auto for small, lbfgs for larger
        "rf": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

def evaluate_models(X, y, models=None, cv_splits: int = 5) -> Dict[str, Tuple[float, float]]:
    """
    Cross-validate each model and return mean/std F1 scores.
    """
    if models is None:
        models = candidate_models()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        results[name] = (float(np.mean(scores)), float(np.std(scores)))
    return results
