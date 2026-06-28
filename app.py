import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time
import pkg_resources


st.set_page_config(
    page_title="Traffic Demand Predictor", page_icon="🚗", layout="centered"
)


@st.cache_resource
def load_models():

    xgb = joblib.load("xgboost_tuned.joblib")
    lgbm = joblib.load("lightgbm_tuned.joblib")
    return xgb, lgbm


xgb_model, lgbm_model = load_models()


def preprocess_input(
    temp, clouds_all, weather_main, selected_hour, selected_date, weather_impact_score
):
    
    df = pd.DataFrame(index=[0])

    df["temp"] = temp
    df["clouds_all"] = clouds_all

    weather_map = {
        "Clear": 0,
        "Clouds": 1,
        "Rain": 2,
        "Snow": 3,
        "Mist": 4,
        "Thunderstorm": 5,
        "Drizzle": 6,
        "Fog": 7,
    }

    df["weather_encoded"] = weather_map.get(weather_main, 0)

    df["hour_sin"] = np.sin(2 * np.pi * selected_hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * selected_hour / 24)

    df["peak_hour_flag"] = 1 if selected_hour in [7, 8, 9, 16, 17, 18] else 0

    day_of_week = selected_date.weekday()  # Monday=0, Sunday=6
    df["weekend_flag"] = 1 if day_of_week >= 5 else 0
    df["working_day"] = 1 if day_of_week < 5 else 0

    df["weather_impact_score"] = weather_impact_score

    final_columns = ['temp', 'rain_1h', 'clouds_all', 'hour_sin', 'hour_cos', 
                     'weather_main_new', 'dayofweek_new', 'month_new']

    for col in final_columns:
        if col not in df.columns:
            df[col] = 0

    return df[final_columns]


st.title("🚗Traffic Predictor")
st.markdown(
    "Predict hourly traffic volume using an Ensemble Machine Learning model (LightGBM + XGBoost)."
)
st.markdown("---")

st.sidebar.header("Input Parameters")


selected_date = st.sidebar.date_input("Select Date", datetime.today())
selected_time = st.sidebar.time_input("Select Time", time(12,0))
selected_hour = selected_time.hour


st.sidebar.subheader("Weather Conditions")
temp = st.sidebar.slider(
    "Temperature (Kelvin)", min_value=240.0, max_value=310.0, value=290.0, step=1.0
)
clouds_all = st.sidebar.slider(
    "Cloud Cover (%)", min_value=0, max_value=100, value=40, step=1
)
weather_main = st.sidebar.selectbox(
    "Main Weather",
    ["Clear", "Clouds", "Rain", "Snow", "Mist", "Thunderstorm", "Drizzle", "Fog"],
)


weather_impact_score = st.sidebar.number_input(
    "Weather Impact Score", min_value=0.0, max_value=10.0, value=5.0, step=0.1
)


if st.button("Predict Traffic Volume", type="primary"):

    X_user = preprocess_input(
        temp,
        clouds_all,
        weather_main,
        selected_hour,
        selected_date,
        weather_impact_score,
    )

    pred_xgb = xgb_model.predict(X_user)[0]
    pred_lgbm = lgbm_model.predict(X_user)[0]

    final_prediction = (0.55 * pred_lgbm) + (0.45 * pred_xgb)

    st.success(f"### Predicted Traffic Volume: {int(final_prediction)} vehicles/hour")

    with st.expander("View Model Breakdown"):
        st.write(f"**LightGBM (55% Weight):** {int(pred_lgbm)} vehicles")
        st.write(f"**XGBoost (45% Weight):** {int(pred_xgb)} vehicles")

        st.write("**Processed Features Array:**")
        st.dataframe(X_user)
