# src/feature_selection.py

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

def make_selector():
    """
    Creates a feature selector using Logistic Regression with L1 penalty.
    Features with low importance are dropped automatically.
    """
    base = LogisticRegression(
        penalty="l1", solver="liblinear", max_iter=500
    )
    selector = SelectFromModel(estimator=base, threshold="median")
    return selector
