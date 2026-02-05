import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingestion.db import get_connection


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].sum() / max(y_true.sum(), 1))


def main():
    con = get_connection()
    df = con.execute("SELECT * FROM stg.transactions").fetchdf()
    con.close()

    y = df["is_fraud"].astype(int).to_numpy()
    X = df.drop(columns=["is_fraud"])

    # time-aware split (no shuffle)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")),
        ]
    )

    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")  # much more meaningful for imbalance

    for k in [100, 500, 1000, 5000]:
        p = precision_at_k(y_test, scores, k)
        r = recall_at_k(y_test, scores, k)
        print(f"Precision@{k}: {p:.4f} | Recall@{k}: {r:.4f}")

    # Example: choose threshold by top-K instead of fixed 0.5
    k = 500
    threshold = np.sort(scores)[-k]
    flagged = (scores >= threshold).sum()
    print(f"\nUsing top-{k} threshold ≈ {threshold:.6f} -> flagged={flagged}")

if __name__ == "__main__":
    main()
