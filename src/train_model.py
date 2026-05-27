import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data_loading import load_raw_data
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERICAL_FEATURES,
    CarFeatureEngineer,
)
from src.preprocessing import clean_training_data

TARGET_COLUMN = "price"
TRAIN_SAMPLE_SIZE = 60_000


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=40,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_model_pipeline(regressor) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        steps=[
            ("features", CarFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", regressor),
        ]
    )

    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def get_candidate_models() -> dict:
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_select_model():
    raw_data = load_raw_data()
    data = clean_training_data(raw_data)

    if len(data) > TRAIN_SAMPLE_SIZE:
        data = data.sample(TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = {}
    trained_models = {}

    for model_name, regressor in get_candidate_models().items():
        print(f"Training {model_name}...")
        model = build_model_pipeline(regressor)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[model_name] = evaluate_predictions(y_test, predictions)
        trained_models[model_name] = model
        print(model_name, results[model_name])

    best_model_name = min(results, key=lambda name: results[name]["MAE"])
    best_model = trained_models[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "model_name": best_model_name,
        **results[best_model_name],
        "all_models": results,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": TARGET_COLUMN,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "used_sample_size": int(len(data)),
        "training_datetime": datetime.now().isoformat(timespec="seconds"),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Best model: {best_model_name}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")

    return best_model, metrics


if __name__ == "__main__":
    train_and_select_model()
