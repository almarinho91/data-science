# Customer Churn Risk Pipeline

Small end-to-end data science project focused on predicting customer churn using a public telecom dataset.

The goal of this project is to practice and demonstrate a realistic data science workflow, from raw data to a trained and evaluated model, including experiment tracking and model interpretation.

---

## Problem

Customer churn refers to customers leaving a service.

In subscription-based businesses such as telecom, SaaS, or banking, churn has a direct financial impact. Predicting which customers are likely to churn allows companies to take proactive retention actions.

This project builds a binary classification model that estimates the probability of customer churn.

---

## Dataset

Public **Telco Customer Churn** dataset.

- Rows: 7,043 customers
- Target variable: `Churn` (Yes / No)
- Churn rate: ~26%

Features include:
- Customer tenure
- Monthly and total charges
- Contract type
- Internet service
- Support services
- Demographics

---

## Modeling Approach

### Baseline Model

A Logistic Regression model was trained using a Scikit-Learn pipeline:

- Numerical features → StandardScaler
- Categorical features → OneHotEncoder
- Classifier → LogisticRegression

Baseline performance:
- ROC-AUC ≈ 0.84

---

### Evaluation Metrics

Because churn data is imbalanced, accuracy alone is not sufficient.

The model is evaluated using:
- ROC-AUC
- Precision
- Recall
- F1-score

---

## Experiment Tracking (MLflow)

MLflow is used to:
- Track experiments
- Log model parameters
- Store evaluation metrics
- Save trained models as artifacts

---

## Hyperparameter Tuning

Optuna is used to tune Logistic Regression hyperparameters:
- Regularization strength (`C`)
- Solver choice

Each trial is logged to MLflow as a separate run, allowing performance comparison across experiments.

---

## Model Interpretation

Logistic Regression coefficients are extracted and analyzed to understand which features contribute most to churn risk.

Examples of influential features:
- Tenure
- Contract type
- Monthly charges
- Internet service type

This interpretability is critical in business settings where models must be explainable.
