from datetime import datetime

import pandas as pd


COLUMN_RENAME_MAP = {
    "Price": "price",
    "Currency": "currency",
    "Condition": "condition",
    "Vehicle_brand": "brand",
    "Vehicle_model": "model",
    "Production_year": "production_year",
    "Mileage_km": "mileage",
    "Power_HP": "engine_power",
    "Displacement_cm3": "engine_capacity",
    "Fuel_type": "fuel_type",
    "Drive": "drive",
    "Transmission": "gearbox",
    "Type": "body_type",
    "Doors_number": "doors_number",
    "Colour": "colour",
    "Origin_country": "origin_country",
    "First_owner": "first_owner",
    "Offer_location": "location",
}


def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(columns=COLUMN_RENAME_MAP).copy()


def clean_training_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_column_names(data)
    cleaned = cleaned.drop_duplicates()

    if "currency" in cleaned.columns:
        cleaned = cleaned[cleaned["currency"].fillna("PLN").eq("PLN")]

    numeric_columns = [
        "price",
        "production_year",
        "mileage",
        "engine_power",
        "engine_capacity",
        "doors_number",
    ]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    current_year = datetime.now().year

    # These ranges remove obvious mistakes while keeping normal used car offers.
    cleaned = cleaned[cleaned["price"].between(1_000, 2_000_000, inclusive="both")]
    cleaned = cleaned[cleaned["production_year"].between(1990, current_year + 1, inclusive="both")]
    cleaned = cleaned[cleaned["mileage"].between(0, 1_000_000, inclusive="both") | cleaned["mileage"].isna()]
    cleaned = cleaned[cleaned["engine_power"].between(20, 1_000, inclusive="both") | cleaned["engine_power"].isna()]
    cleaned = cleaned[cleaned["engine_capacity"].between(500, 8_500, inclusive="both") | cleaned["engine_capacity"].isna()]

    text_columns = cleaned.select_dtypes(include="object").columns
    for column in text_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip().replace({"nan": pd.NA, "None": pd.NA})

    return cleaned.reset_index(drop=True)
