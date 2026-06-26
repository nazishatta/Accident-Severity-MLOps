import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("models/lightgbm_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")


class AccidentFeatures(BaseModel):
    Source: str = Field(default="Source2")
    Start_Lat: float = Field(default=39.86)
    Start_Lng: float = Field(default=-84.05)
    Distance_mi: float = Field(default=0.5)
    State: str = Field(default="OH")
    Timezone: str = Field(default="US/Eastern")
    Temperature_F: float = Field(default=60)
    Humidity_pct: float = Field(default=70)
    Pressure_in: float = Field(default=29.9)
    Visibility_mi: float = Field(default=10)
    Wind_Direction: str = Field(default="Calm")
    Wind_Speed_mph: float = Field(default=5)
    Precipitation_in: float = Field(default=0)
    Weather_Condition: str = Field(default="Clear")
    Amenity: int = Field(default=0)
    Bump: int = Field(default=0)
    Crossing: int = Field(default=1)
    Give_Way: int = Field(default=0)
    Junction: int = Field(default=0)
    No_Exit: int = Field(default=0)
    Railway: int = Field(default=0)
    Roundabout: int = Field(default=0)
    Station: int = Field(default=0)
    Stop: int = Field(default=0)
    Traffic_Calming: int = Field(default=0)
    Traffic_Signal: int = Field(default=1)
    Sunrise_Sunset: str = Field(default="Day")
    start_year: int = Field(default=2023)
    is_weekend: int = Field(default=0)
    is_rush_hour: int = Field(default=1)
    duration_min: float = Field(default=30)
    hour_sin: float = Field(default=0.5)
    hour_cos: float = Field(default=0.5)
    dayofweek_sin: float = Field(default=0.5)
    dayofweek_cos: float = Field(default=0.5)
    month_sin: float = Field(default=0.5)
    month_cos: float = Field(default=0.5)


@app.get("/")
def read_root():
    return {"message": "Accident Severity API is running"}


@app.post("/predict")
def predict(features: AccidentFeatures):
    f = features.dict()

    for col in ["Source", "State", "Timezone", "Wind_Direction", "Weather_Condition", "Sunrise_Sunset"]:
        f[col] = encoders[col].transform([f[col]])[0]

    order = [
        "Source", "Start_Lat", "Start_Lng", "Distance_mi", "State", "Timezone",
        "Temperature_F", "Humidity_pct", "Pressure_in", "Visibility_mi",
        "Wind_Direction", "Wind_Speed_mph", "Precipitation_in", "Weather_Condition",
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
        "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
        "Sunrise_Sunset", "start_year", "is_weekend", "is_rush_hour", "duration_min",
        "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos"
    ]
    data = np.array([[f[col] for col in order]])

    prediction = model.predict(data)
    return {"predicted_severity": int(prediction[0])}
