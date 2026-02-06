import json
import numpy as np
import pandas as pd
from pathlib import Path


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingestion.db import get_connection

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"


def precision_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean())


def main():
    con = get_connection()
    df = con.execute("SELECT * FROM features.transactions_features").fetchdf()
    con.close()

    y = df["is_fraud"].astype(int).to_numpy()

    drop_cols = [
        "is_fraud",
        "ts",
        "amount_rollmean_24h",
        "amount_rollstd_24h",
    ]

    X = df.drop(columns=drop_cols)

    feature_names = X.columns.tolist()

    META_PATH = MODEL_DIR / "feature_cols.json"
    META_PATH.write_text(json.dumps(feature_names), encoding="utf-8")
    print(f"Feature columns saved to {META_PATH}")

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    for k in [100, 500]:
        print(f"Precision@{k}: {precision_at_k(y_test, scores, k):.4f}")

    # feature importance (logistic regression coefficients)
    coefs = pipe.named_steps["clf"].coef_[0]
    importance = pd.DataFrame(
        {"feature": feature_names, "coef": coefs}
    ).assign(abs_coef=lambda d: d["coef"].abs()).sort_values("abs_coef", ascending=False)

    print("\nTop 15 features:")
    print(importance.head(15))

    # save model
    import joblib
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
