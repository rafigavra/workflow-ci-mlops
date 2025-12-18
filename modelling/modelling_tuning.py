import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# Load Dataset (PREPROCESSED)
# =========================
data = pd.read_csv("preprocessing/titanic_preprocessed.csv")

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Hyperparameter Tuning
# =========================
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# =========================
# Manual MLflow Logging
# =========================
mlflow.set_experiment("Titanic-Classification-Tuning")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(grid.best_params_)

    # Predict
    y_pred = best_model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
