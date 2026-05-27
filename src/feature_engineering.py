from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing import normalize_column_names


PREMIUM_BRANDS = {
    "Audi",
    "BMW",
    "Mercedes-Benz",
    "Lexus",
    "Porsche",
    "Volvo",
    "Tesla",
    "Jaguar",
    "Land Rover",
    "Infiniti",
    "Maserati",
}

NUMERICAL_FEATURES = [
    "production_year",
    "mileage",
    "engine_power",
    "engine_capacity_l",
    "doors_number",
    "car_age",
    "mileage_per_year",
    "engine_power_per_liter",
    "is_premium_brand",
    "is_new_car",
    "is_high_mileage",
]

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "brand_model",
    "fuel_type",
    "gearbox",
    "body_type",
    "drive",
    "condition",
    "colour",
    "origin_country",
    "first_owner",
    "simplified_location",
]

ENGINEERED_FEATURES = [
    "engine_capacity_l",
    "car_age",
    "mileage_per_year",
    "engine_power_per_liter",
    "is_premium_brand",
    "is_new_car",
    "is_high_mileage",
    "simplified_location",
    "brand_model",
]


def _extract_region(location: object) -> object:
    if pd.isna(location):
        return pd.NA

    text = str(location)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    polish_regions = {
        "Dolnośląskie", "Kujawsko-pomorskie", "Lubelskie", "Lubuskie",
        "Łódzkie", "Małopolskie", "Mazowieckie", "Opolskie",
        "Podkarpackie", "Podlaskie", "Pomorskie", "Śląskie",
        "Świętokrzyskie", "Warmińsko-mazurskie", "Wielkopolskie", "Zachodniopomorskie",
    }
    for part in parts:
        clean_part = part.replace(" (Polska)", "").strip()
        if clean_part in polish_regions:
            return clean_part

    if len(parts) >= 2:
        return parts[1].replace(" (Polska)", "").strip()
    return parts[0].replace(" (Polska)", "").strip() if parts else np.nan


class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, current_year: int | None = None, top_brand_models: int = 80):
        self.current_year = current_year
        self.top_brand_models = top_brand_models

    def fit(self, X, y=None):
        data = normalize_column_names(pd.DataFrame(X))
        if {"brand", "model"}.issubset(data.columns):
            combinations = (
                data["brand"].fillna("Unknown") + " " + data["model"].fillna("Unknown")
            )
            self.frequent_brand_models_ = set(
                combinations.value_counts().head(self.top_brand_models).index
            )
        else:
            self.frequent_brand_models_ = set()
        return self

    def transform(self, X):
        data = normalize_column_names(pd.DataFrame(X))
        current_year = self.current_year or datetime.now().year

        for column in ["production_year", "mileage", "engine_power", "engine_capacity", "doors_number"]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
            else:
                data[column] = np.nan

        for column in CATEGORICAL_FEATURES:
            if column not in data.columns and column not in {"brand_model", "simplified_location"}:
                data[column] = pd.NA

        data["engine_capacity_l"] = data["engine_capacity"] / 1000
        small_capacity = data["engine_capacity"].between(0.3, 10, inclusive="both")
        data.loc[small_capacity, "engine_capacity_l"] = data.loc[small_capacity, "engine_capacity"]

        data["car_age"] = (current_year - data["production_year"]).clip(lower=0)
        age_for_ratio = data["car_age"].clip(lower=1)
        data["mileage_per_year"] = data["mileage"] / age_for_ratio
        data["engine_power_per_liter"] = data["engine_power"] / data["engine_capacity_l"]

        data["is_premium_brand"] = data["brand"].isin(PREMIUM_BRANDS).astype(int)
        data["is_new_car"] = (data["car_age"] <= 1).astype(int)
        data["is_high_mileage"] = (data["mileage"] > 200_000).astype(int)

        if "region" in data.columns:
            data["simplified_location"] = data["region"]
        elif "location" in data.columns:
            data["simplified_location"] = data["location"].apply(_extract_region)
        else:
            data["simplified_location"] = pd.NA

        brand_model = data["brand"].fillna("Unknown") + " " + data["model"].fillna("Unknown")
        data["brand_model"] = np.where(
            brand_model.isin(self.frequent_brand_models_), brand_model, "Other"
        )

        return data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].replace({pd.NA: np.nan})
