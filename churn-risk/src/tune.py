import optuna
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


DATA_PATH = "data/raw/churn.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])

    # fix TotalCharges
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X = X.fillna(0)

    return X, y


def build_pipeline(C, solver):
    # We rebuild preprocessing each time for simplicity
    X, y = load_data()

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=2000,
        class_weight="balanced",
    )

    pipe = Pipeline([("prep", pre), ("clf", clf)])
    return pipe, X, y


def objective(trial):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

    model, X, y = build_pipeline(C, solver)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # One MLflow run per trial (nested)
    with mlflow.start_run(nested=True):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 2000)
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        mlflow.log_metric("roc_auc", float(auc))

    return auc


def main():
    mlflow.set_experiment("churn-optuna")

    # Parent run that groups all trials
    with mlflow.start_run(run_name="optuna_sweep"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        # Log the best result to the parent run too
        mlflow.log_metric("best_roc_auc", float(study.best_value))
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        print("Best params:", study.best_params)
        print("Best ROC-AUC:", study.best_value)


if __name__ == "__main__":
    main()
