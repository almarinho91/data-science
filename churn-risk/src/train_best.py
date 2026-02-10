import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


DATA_PATH = "data/raw/churn.csv"
BEST_C = 0.6135273012774716
BEST_SOLVER = "liblinear"


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])

    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X = X.fillna(0)
    return X, y


def build_pipeline():
    X, _ = load_data()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = LogisticRegression(
        C=BEST_C,
        solver=BEST_SOLVER,
        max_iter=2000,
        class_weight="balanced",
    )

    return Pipeline([("prep", pre), ("clf", clf)])


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_pipeline()

    mlflow.set_experiment("churn-best-model")

    Path("models").mkdir(exist_ok=True)

    with mlflow.start_run(run_name="logreg_best"):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 2000)
        mlflow.log_param("C", BEST_C)
        mlflow.log_param("solver", BEST_SOLVER)

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        print("ROC-AUC:", round(auc, 4))
        mlflow.log_metric("roc_auc", float(auc))

        rep = classification_report(y_test, probs > 0.5, output_dict=True)
        mlflow.log_metric("precision_churn", float(rep["1"]["precision"]))
        mlflow.log_metric("recall_churn", float(rep["1"]["recall"]))
        mlflow.log_metric("f1_churn", float(rep["1"]["f1-score"]))

        print(classification_report(y_test, probs > 0.5))

        # log model
        mlflow.sklearn.log_model(model, "model")

        print("Logged tuned model to MLflow.")


if __name__ == "__main__":
    main()
