import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict_car_price(car_data: dict) -> float:
    model = load_model()
    car_df = pd.DataFrame([car_data])
    prediction = model.predict(car_df)[0]
    return float(round(prediction, 2))


if __name__ == "__main__":
    sample_car = {
        "brand": "Toyota",
        "model": "Corolla",
        "production_year": 2018,
        "mileage": 90_000,
        "fuel_type": "Gasoline",
        "gearbox": "Manual",
        "engine_capacity": 1600,
        "engine_power": 132,
        "body_type": "sedan",
        "drive": "Front wheels",
        "condition": "Used",
        "doors_number": 5,
        "colour": "gray",
        "origin_country": "Poland",
        "first_owner": "Yes",
        "region": "Mazowieckie",
    }

    predicted_price = predict_car_price(sample_car)
    print(f"Predicted price: {predicted_price:,.2f} PLN")
