# AutoML Credit Default Prediction

This project predicts **credit card default risk** using machine learning.  
We build a pipeline for preprocessing, feature selection, hyperparameter tuning, and evaluation.  
The goal is to demonstrate a reproducible ML workflow with saved models and metrics.

---

## 📂 Project Structure
automl-credit-default/
│
├── data/ # Dataset (UCI Credit Card dataset)
├── notebooks/ # Jupyter notebooks (EDA, experiments)
├── src/ # Source code (preprocessing, feature selection, tuning, evaluation)
├── results/ # Saved models, metrics, plots
│ ├── final_pipeline.joblib
│ ├── metrics.json
│ ├── confusion_matrix.png
│ └── roc_curve.png
├── .gitignore
├── requirements.txt
└── README.md

📊 Results (Test Set)

From results/metrics.json:

Baseline RF (cross-val F1): 0.474

Final Test F1: 0.460

Precision: 0.630

Recall: 0.362

ROC AUC: 0.759

Test size: 6000