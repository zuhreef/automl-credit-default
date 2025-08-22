# src/preprocessing.py

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Target column name
TARGET = "default.payment.next.month"

def train_test_split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataframe into train and test sets.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a preprocessing pipeline.
    Currently: scales all numeric features.
    """
    num_features = X.columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features)
        ]
    )
    return preprocessor
