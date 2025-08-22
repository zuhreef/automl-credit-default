import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.show()

def plot_roc(model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve"); plt.tight_layout(); plt.show()
    try:
        proba = model.predict_proba(X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass

