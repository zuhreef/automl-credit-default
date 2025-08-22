import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

def tune_random_forest(X, y, n_trials: int = 30, random_state: int = 42):
    """
    X, y should be the transformed training data (after preprocessor + selector).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params)
        score = cross_val_score(clf, X, y, cv=cv, scoring=scorer, n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study
