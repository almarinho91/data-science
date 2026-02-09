import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

DATA = "data/raw/churn.csv"


def main():
    df = pd.read_csv(DATA)

    print("Rows:", len(df))
    print(df["Churn"].value_counts(normalize=True))

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])

    # fix TotalCharges
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X = X.fillna(0)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", pre),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)
    print("ROC-AUC:", round(auc, 4))
    print(classification_report(y_test, preds > 0.5))


if __name__ == "__main__":
    main()
