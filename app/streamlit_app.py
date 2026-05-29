from pathlib import Path
import sys

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.car_options import load_car_options
from src.config import MODEL_PATH


st.set_page_config(
    page_title="Used Car Price Prediction PL",
    page_icon="🚗",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def round_price(price: float) -> int:
    return int(round(price / 100) * 100)


def predict_price(car_data: dict) -> int:
    import pandas as pd

    model = load_model()
    input_data = pd.DataFrame([car_data])
    prediction = model.predict(input_data)[0]
    return round_price(float(prediction))


def render_home() -> None:
    st.title("Used Car Price Prediction PL")
    st.write(
        "This app estimates the market price of a used car in Poland based on "
        "details from historical car sale offers."
    )

    step_1, step_2, step_3 = st.columns(3)
    with step_1:
        st.markdown("**1. Enter car details**")
        st.caption("Choose the most important information about the car.")
    with step_2:
        st.markdown("**2. Run prediction**")
        st.caption("The trained model processes the input.")
    with step_3:
        st.markdown("**3. Get price**")
        st.caption("See the estimated price in PLN.")

    with st.expander("How to use this app?"):
        st.write(
            "Fill in realistic car information and click the prediction button. "
            "The result is an estimate, not a guaranteed sale price."
        )


def render_car_form() -> tuple[dict, bool]:
    options = load_car_options()

    st.header("Car details")
    st.caption("Choose values similar to a real Polish used car offer.")

    with st.form("car_prediction_form"):
        left, right = st.columns(2)

        with left:
            brand = st.selectbox("Brand", options["brands"], index=options["brands"].index("Toyota") if "Toyota" in options["brands"] else 0)
            model_options = options["brand_models"].get(brand, ["Other"])
            model = st.selectbox("Model", model_options)
            production_year = st.number_input(
                "Production year", min_value=1990, max_value=2027, value=2018, step=1
            )
            mileage = st.number_input(
                "Mileage [km]", min_value=0, max_value=1_000_000, value=90_000, step=5_000
            )
            condition = st.selectbox("Condition", options["conditions"])

        with right:
            fuel_type = st.selectbox("Fuel type", options["fuel_types"])
            gearbox = st.selectbox("Gearbox", options["gearboxes"])
            body_type = st.selectbox("Body type", options["body_types"])
            engine_capacity_l = st.number_input(
                "Engine capacity [liters]", min_value=0.5, max_value=8.5, value=1.6, step=0.1
            )
            engine_power = st.number_input(
                "Engine power [HP]", min_value=20, max_value=1000, value=132, step=5
            )

        details_left, details_right = st.columns(2)
        with details_left:
            region = st.selectbox("Region", options["regions"])
            drive = st.selectbox("Drive", options["drives"])
        with details_right:
            first_owner = st.checkbox("First owner", value=True)
            doors_number = st.selectbox("Doors", [3, 4, 5], index=2)

        submitted = st.form_submit_button("Predict price", use_container_width=True)

    car_data = {
        "brand": brand,
        "model": model,
        "production_year": int(production_year),
        "mileage": int(mileage),
        "fuel_type": fuel_type,
        "gearbox": gearbox,
        "engine_capacity": float(engine_capacity_l) * 1000,
        "engine_power": int(engine_power),
        "body_type": body_type,
        "drive": drive,
        "condition": condition,
        "doors_number": int(doors_number),
        "colour": "gray",
        "origin_country": "Poland",
        "first_owner": "Yes" if first_owner else None,
        "region": region,
    }

    return car_data, submitted


def render_prediction_result(car_data: dict) -> None:
    try:
        predicted_price = predict_price(car_data)
    except FileNotFoundError:
        st.error(
            "The trained model file is missing. Please train the model first with "
            "`python3 -m src.train_model`."
        )
        return
    except Exception as error:
        st.error("Prediction failed. Please check the entered car details and try again.")
        with st.expander("Technical details"):
            st.code(str(error))
        return

    st.success("Prediction ready")
    st.metric("Estimated car price", f"{predicted_price:,.0f} PLN".replace(",", " "))
    st.caption(
        "This is an estimated price based on historical car offers from Poland. "
        "Real sale prices can be different because of car condition, equipment, negotiation, and market changes."
    )


def main() -> None:
    render_home()
    st.divider()
    car_data, submitted = render_car_form()

    if submitted:
        st.divider()
        render_prediction_result(car_data)


if __name__ == "__main__":
    main()
