import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingestion.db import get_connection


def precision_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean())


def recall_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].sum() / max(y_true.sum(), 1))


def main():
    con = get_connection()

    df = con.execute("""
        SELECT *
        FROM features.transactions_features
    """).fetchdf()

    con.close()

    y = df["is_fraud"].astype(int).to_numpy()

    drop_cols = [
        "is_fraud",
        "ts",
        "amount_rollmean_24h",
        "amount_rollstd_24h",
    ]

    X = df.drop(columns=drop_cols)

    # time-aware split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    for k in [100, 500, 1000]:
        print(
            f"Precision@{k}: {precision_at_k(y_test, scores, k):.4f} | "
            f"Recall@{k}: {recall_at_k(y_test, scores, k):.4f}"
        )


if __name__ == "__main__":
    main()
