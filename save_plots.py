# save_plots.py
import pandas as pd
from joblib import load
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

PIPE = "results/final_pipeline.joblib"
DATA = "data/credit_clean.csv"

pipe = load(PIPE)
df = pd.read_csv(DATA)
X = df.drop(columns=["default.payment.next.month"])
y = df["default.payment.next.month"]

# Confusion Matrix
cm = confusion_matrix(y, pipe.predict(X))
fig_cm, ax_cm = plt.subplots(figsize=(4,4))
ConfusionMatrixDisplay(cm).plot(ax=ax_cm, colorbar=False)
ax_cm.set_title("Confusion Matrix")
fig_cm.tight_layout()
fig_cm.savefig("results/confusion_matrix.png", dpi=150)

# ROC Curve
fig_roc, ax_roc = plt.subplots()
RocCurveDisplay.from_estimator(pipe, X, y, ax=ax_roc)
ax_roc.set_title("ROC Curve")
fig_roc.tight_layout()
fig_roc.savefig("results/roc_curve.png", dpi=150)

print("Saved plots to results/")
