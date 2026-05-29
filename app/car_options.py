from functools import lru_cache
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loading import load_raw_data
from src.preprocessing import clean_training_data


FALLBACK_OPTIONS = {
    "brands": ["Toyota", "Volkswagen", "BMW", "Audi", "Opel", "Ford", "Mercedes-Benz", "Other"],
    "brand_models": {
        "Toyota": ["Corolla", "Yaris", "Auris", "Avensis", "Other"],
        "Volkswagen": ["Golf", "Passat", "Polo", "Tiguan", "Other"],
        "BMW": ["Seria 3", "Seria 5", "X3", "X5", "Other"],
        "Audi": ["A4", "A6", "A3", "Q5", "Other"],
        "Opel": ["Astra", "Corsa", "Insignia", "Vectra", "Other"],
        "Ford": ["Focus", "Mondeo", "Fiesta", "Kuga", "Other"],
        "Mercedes-Benz": ["Klasa C", "Klasa E", "GLC", "Other"],
        "Other": ["Other"],
    },
    "fuel_types": ["Gasoline", "Diesel", "Hybrid", "Gasoline + LPG", "Electric"],
    "gearboxes": ["Manual", "Automatic"],
    "body_types": ["SUV", "station_wagon", "sedan", "compact", "city_cars", "minivan", "coupe"],
    "drives": ["Front wheels", "Rear wheels", "4x4 (permanent)", "4x4 (attached automatically)"],
    "conditions": ["Used", "New"],
    "colours": ["black", "gray", "silver", "white", "blue", "red", "other"],
    "origin_countries": ["Poland", "Germany", "France", "Belgium", "United States", "Other"],
}

REGIONS = [
    "Mazowieckie",
    "Śląskie",
    "Wielkopolskie",
    "Małopolskie",
    "Dolnośląskie",
    "Pomorskie",
    "Łódzkie",
    "Kujawsko-pomorskie",
    "Lubelskie",
    "Podkarpackie",
    "Zachodniopomorskie",
    "Warmińsko-mazurskie",
    "Świętokrzyskie",
    "Podlaskie",
    "Lubuskie",
    "Opolskie",
]


def _top_values(data: pd.DataFrame, column: str, limit: int) -> list[str]:
    if column not in data.columns:
        return []
    values = data[column].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return values.value_counts().head(limit).index.tolist()


@lru_cache(maxsize=1)
def load_car_options() -> dict:
    try:
        data = clean_training_data(load_raw_data())
    except Exception:
        return {**FALLBACK_OPTIONS, "regions": REGIONS}

    brands = _top_values(data, "brand", 40)
    if "Other" not in brands:
        brands.append("Other")

    brand_models = {}
    for brand in brands:
        if brand == "Other":
            brand_models[brand] = ["Other"]
            continue
        models = _top_values(data[data["brand"].eq(brand)], "model", 30)
        brand_models[brand] = models + (["Other"] if "Other" not in models else [])

    return {
        "brands": brands,
        "brand_models": brand_models,
        "fuel_types": _top_values(data, "fuel_type", 8) or FALLBACK_OPTIONS["fuel_types"],
        "gearboxes": _top_values(data, "gearbox", 4) or FALLBACK_OPTIONS["gearboxes"],
        "body_types": _top_values(data, "body_type", 10) or FALLBACK_OPTIONS["body_types"],
        "drives": _top_values(data, "drive", 6) or FALLBACK_OPTIONS["drives"],
        "conditions": _top_values(data, "condition", 3) or FALLBACK_OPTIONS["conditions"],
        "colours": _top_values(data, "colour", 12) or FALLBACK_OPTIONS["colours"],
        "origin_countries": _top_values(data, "origin_country", 12) or FALLBACK_OPTIONS["origin_countries"],
        "regions": REGIONS,
    }
