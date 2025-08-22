import argparse, pandas as pd
from joblib import load

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with the same feature columns (no target needed).")
    args = p.parse_args()

    pipe = load("results/final_pipeline.joblib")
    X = pd.read_csv(args.csv).drop(columns=["default.payment.next.month"], errors="ignore")
    preds = pipe.predict(X)
    out = X.copy()
    out["prediction"] = preds
    try:
        out["prob_default"] = pipe.predict_proba(X)[:,1]
    except Exception:
        pass
    out.to_csv("results/predictions.csv", index=False)
    print("Wrote results/predictions.csv")

if __name__ == "__main__":
    main()
