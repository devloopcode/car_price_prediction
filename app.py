import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="UK Used Car Price Estimator", page_icon="🚗")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "price_model.joblib")


@st.cache_data
def load_options():
    with open(MODEL_DIR / "options.json") as f:
        return json.load(f)


pipe = load_model()
opt = load_options()

st.title("🚗 UK Used Car Price Estimator")
st.write(
    "Enter the details of a used car and get an estimated price in £. "
    "The model was trained on about 97,000 UK listings (2020 and earlier)."
)

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", sorted(opt["brands"].keys()))
    model_name = st.selectbox("Model", opt["brands"][brand])
    year = st.slider("Year", opt["year_min"], opt["year_max"], opt["defaults"]["year"])
    mileage = st.number_input("Mileage", min_value=0, max_value=opt["mileage_max"],
                              value=opt["defaults"]["mileage"], step=1000)

with col2:
    fuel = st.selectbox("Fuel type", opt["fuelType"])
    transmission = st.selectbox("Transmission", opt["transmission"])
    engine = st.number_input("Engine size (litres)", min_value=0.0, max_value=7.0,
                             value=float(opt["defaults"]["engineSize"]), step=0.1)
    mpg = st.number_input("MPG", min_value=10.0, max_value=300.0,
                          value=float(opt["defaults"]["mpg"]), step=0.5)

tax = st.number_input("Road tax (£ per year)", min_value=0, max_value=600,
                      value=int(opt["defaults"]["tax"]), step=5)

if st.button("Estimate price", type="primary"):
    row = pd.DataFrame([{
        "model": model_name,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel,
        "tax": float(tax),
        "mpg": float(mpg),
        "engineSize": float(engine),
        "brand": brand,
    }])
    price = float(pipe.predict(row)[0])
    price = max(price, 0.0)
    st.success(f"Estimated price: **£{price:,.0f}**")
    st.caption(
        f"On cars it had never seen, the model was off by about £{opt['test_mae']:,} on average, "
        "so treat this as a rough guide, not a valuation."
    )

with st.expander("About this project"):
    st.write(
        "Pipeline: one-hot encoding for text columns + a tree-based regressor, "
        f"model: **{opt['model_name']}**. Test R²: {opt['test_r2']:.3f}. "
        "Predictions for rare or unusual combinations (for example, a very high "
        "mileage on a recent car) are less reliable."
    )