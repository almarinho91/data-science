import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from ingestion.db import get_connection


def main():
    con = get_connection()

    df = con.execute("""
        SELECT *
        FROM stg.transactions
    """).fetchdf()

    con.close()

    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])

    # Time-aware split (no shuffle)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)

    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds > 0.5))


if __name__ == "__main__":
    main()
