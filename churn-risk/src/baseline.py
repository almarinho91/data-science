import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


DATA = "data/raw/churn.csv"


def main():
    df = pd.read_csv(DATA)

    print("Rows:", len(df))
    print(df["Churn"].value_counts(normalize=True))

    # target
    y = (df["Churn"] == "Yes").astype(int)

    # features (drop target + ID)
    X = df.drop(columns=["Churn", "customerID"])

    # fix TotalCharges (has blanks)
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X = X.fillna(0)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", pre),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- MLflow tracking ----
    mlflow.set_experiment("churn-baseline")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    with mlflow.start_run():
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 2000)

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, preds)
        print("ROC-AUC:", round(auc, 4))
        mlflow.log_metric("roc_auc", float(auc))

        # classification report
        report = classification_report(y_test, preds > 0.5, output_dict=True)
        # log churn class metrics
        mlflow.log_metric("precision_churn", float(report["1"]["precision"]))
        mlflow.log_metric("recall_churn", float(report["1"]["recall"]))
        mlflow.log_metric("f1_churn", float(report["1"]["f1-score"]))

        print(classification_report(y_test, preds > 0.5))

        # feature importance via coefficients
        feature_names = model.named_steps["prep"].get_feature_names_out()
        coefs = model.named_steps["clf"].coef_[0]

        imp = (
            pd.DataFrame({"feature": feature_names, "coef": coefs})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False)
        )

        print("\nTop 15 features:")
        print(imp.head(15))

        imp_path = models_dir / "feature_importance.csv"
        imp.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path))

        # log the full pipeline as an MLflow model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("\nLogged run to MLflow. Artifacts saved.")


if __name__ == "__main__":
    main()
