import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import FIGURES_DIR, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data_loading import load_raw_data
from src.preprocessing import clean_training_data
from src.train_model import TARGET_COLUMN, TRAIN_SAMPLE_SIZE


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_test_split():
    data = clean_training_data(load_raw_data())
    if len(data) > TRAIN_SAMPLE_SIZE:
        data = data.sample(TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def save_evaluation_plots(y_true, y_pred) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({"actual_price": y_true, "predicted_price": y_pred})
    sample = results.sample(min(3000, len(results)), random_state=RANDOM_STATE)

    plt.figure(figsize=(7, 6))
    plt.scatter(sample["actual_price"], sample["predicted_price"], alpha=0.25)
    max_price = sample[["actual_price", "predicted_price"]].quantile(0.99).max()
    plt.plot([0, max_price], [0, max_price], color="red", linewidth=2)
    plt.xlim(0, max_price)
    plt.ylim(0, max_price)
    plt.xlabel("Actual price [PLN]")
    plt.ylabel("Predicted price [PLN]")
    plt.title("Actual vs predicted prices")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=60)
    plt.xlabel("Prediction error [PLN]")
    plt.ylabel("Number of cars")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals.png", dpi=150)
    plt.close()


def evaluate_saved_model() -> dict:
    model = joblib.load(MODEL_PATH)
    _, X_test, _, y_test = load_test_split()
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    save_evaluation_plots(y_test, predictions)

    print("Evaluation of saved model")
    print(f"MAE:  {metrics['MAE']:.2f} PLN")
    print(f"RMSE: {metrics['RMSE']:.2f} PLN")
    print(f"R2:   {metrics['R2']:.4f}")

    if METRICS_PATH.exists():
        saved_metrics = json.loads(METRICS_PATH.read_text())
        print(f"Saved model name: {saved_metrics.get('model_name')}")

    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
