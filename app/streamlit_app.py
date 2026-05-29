from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


st.set_page_config(
    page_title="Used Car Price Prediction PL",
    page_icon="🚗",
    layout="centered",
)


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


def render_prediction_placeholder() -> None:
    st.header("Car details")
    st.info("The input form will be added in the next step.")


def main() -> None:
    render_home()
    st.divider()
    render_prediction_placeholder()


if __name__ == "__main__":
    main()
