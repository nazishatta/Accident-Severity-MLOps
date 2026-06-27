import streamlit as st
import requests

st.title("US Accident Severity Predictor")
st.write("Enter accident conditions below to predict severity.")

st.header("Location & Weather")
start_lat = st.number_input("Latitude", value=39.86)
start_lng = st.number_input("Longitude", value=-84.05)
state = st.text_input("State (e.g. OH)", value="OH")
temperature = st.number_input("Temperature (F)", value=60.0)
humidity = st.number_input("Humidity (%)", value=70.0)
visibility = st.number_input("Visibility (mi)", value=10.0)
wind_speed = st.number_input("Wind Speed (mph)", value=5.0)
weather_condition = st.text_input("Weather Condition (e.g. Clear)", value="Clear")

if st.button("Predict Severity"):
    payload = {
        "Source": "Source2",
        "Start_Lat": start_lat,
        "Start_Lng": start_lng,
        "Distance_mi": 0.5,
        "State": state,
        "Timezone": "US/Eastern",
        "Temperature_F": temperature,
        "Humidity_pct": humidity,
        "Pressure_in": 29.9,
        "Visibility_mi": visibility,
        "Wind_Direction": "Calm",
        "Wind_Speed_mph": wind_speed,
        "Precipitation_in": 0,
        "Weather_Condition": weather_condition,
        "Amenity": 0, "Bump": 0, "Crossing": 1, "Give_Way": 0,
        "Junction": 0, "No_Exit": 0, "Railway": 0, "Roundabout": 0,
        "Station": 0, "Stop": 0, "Traffic_Calming": 0, "Traffic_Signal": 1,
        "Sunrise_Sunset": "Day",
        "start_year": 2023, "is_weekend": 0, "is_rush_hour": 1,
        "duration_min": 30, "hour_sin": 0.5, "hour_cos": 0.5,
        "dayofweek_sin": 0.5, "dayofweek_cos": 0.5,
        "month_sin": 0.5, "month_cos": 0.5
    }
    response = requests.post("http://localhost:8000/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Severity: {result['predicted_severity']}")
    else:
        st.error(f"Error: {response.status_code}")
