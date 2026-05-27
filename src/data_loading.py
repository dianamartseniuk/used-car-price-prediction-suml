from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR


def find_raw_csv(raw_data_dir: Path = RAW_DATA_DIR) -> Path:
    csv_files = sorted(raw_data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {raw_data_dir}. "
            "Download the Kaggle dataset and place it in data/raw/."
        )

    if len(csv_files) == 1:
        return csv_files[0]

    for csv_file in csv_files:
        sample_columns = pd.read_csv(csv_file, nrows=0).columns
        if {"Price", "Vehicle_brand", "Production_year"}.issubset(sample_columns):
            return csv_file

    raise ValueError(
        "More than one CSV file was found, but none looked like the car ads dataset."
    )


def load_raw_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    csv_path = find_raw_csv(raw_data_dir)
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    data = load_raw_data()
    print(f"Loaded data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
