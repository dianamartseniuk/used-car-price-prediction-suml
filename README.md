Created by:
Diana Martseniuk
Julia Zawadzka

Link to app:  https://ca-usedcarprice-dev-pl.happysea-a9342bdf.polandcentral.azurecontainerapps.io

Input:
- Finding data and topic - Diana Martseniuk
- Documentation - Diana Martseniuk
- ML models - Diana Martseniuk
- Streamlit - Diana Martseniuk
- Docker - Julia Zawadzka
- Cloud - Julia Zawadzka
- CI/CD - Julia Zawadzka
- Terraform - Julia Zawadzka

# Used Car Price Prediction PL

This is a student machine learning project for predicting used car prices in Poland. The problem is regression, because the model predicts a numeric price in PLN.

The model is prepared as a scikit-learn pipeline, so later it can be reused in a Streamlit app without rewriting the preprocessing logic.

## Dataset

Dataset source: https://www.kaggle.com/datasets/bartoszpieniak/poland-cars-for-sale-dataset/data


## Technologies

- Python
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib and seaborn for exploration/evaluation plots
- Jupyter notebook

## Features

The first model uses basic car advertisement information:

- brand and model
- production year
- mileage
- fuel type
- gearbox
- engine capacity
- engine power
- body type
- drive type
- condition
- colour
- origin country
- first owner information
- simplified location

Columns like offer index, exact dates, long feature lists, and exact messy location text are not used directly in the first version.

## Feature Engineering

I added a few simple features that should make sense for used cars:

- `car_age`: newer cars usually cost more
- `mileage_per_year`: separates normal usage from very intensive usage
- `engine_power_per_liter`: rough indicator of engine performance
- `is_premium_brand`: premium brands often keep higher prices
- `is_new_car`: cars up to one year old are different from normal used cars
- `is_high_mileage`: cars over 200,000 km often lose value
- `simplified_location`: uses region instead of exact address text
- `brand_model`: keeps only the most common brand-model combinations

The target price is trained with `log1p(price)` and converted back with `expm1`, because car prices are very skewed.

## Cleaning Rules

The cleaning step removes obvious bad rows:

- only PLN offers are used
- price must be between 1,000 and 2,000,000 PLN
- mileage must be between 0 and 1,000,000 km
- production year must be between 1990 and current year + 1
- engine power must be between 20 and 1,000 HP
- engine capacity must be between 500 and 8,500 cm3

These rules are simple, but they remove unrealistic outliers from the dataset.

## Model Training

The training script compares:

- Ridge
- RandomForestRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor

The final model is selected mainly by MAE, because it is easy to explain as the average prediction error in PLN.

For faster local training, the script uses a random sample of 60,000 cleaned rows.

Current saved model:

```text
RandomForestRegressor
```

Evaluation on the test split:

| Metric | Value |
| --- | ---: |
| MAE | 9,338.76 PLN |
| RMSE | 25,392.92 PLN |
| R2 | 0.9119 |

The trained model is saved locally as:

```text
models/car_price_model.joblib
```

The metrics are saved as:

```text
models/metrics.json
```

## Project Structure

```text
used-car-price-prediction-suml/
  app/
  car_options.py
  streamlit_app.py
  data/
    raw/
    processed/
  notebooks/
    01_data_exploration.ipynb
  src/
    __init__.py
    config.py
    data_loading.py
    preprocessing.py
    feature_engineering.py
    train_model.py
    evaluate_model.py
    predict.py
  models/
    car_price_model.pkl
    metrics.json
  reports/
    figures/
  tests/
  terraform/
    main.tf
    providers.rf 
    variables.tf
  requirements.txt
  README.md
  .gitignore
  .dockerignore
  Dockerfile
  azure-pipelines.yml
```
