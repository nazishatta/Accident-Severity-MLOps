"""
Data preparation stage for the Accident Severity DVC pipeline.
Source logic: US_Accident_version4.ipynb, Cells 15, 18, 24, 33.
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

RAW_PATH = params["data"]["raw_path"]
PROCESSED_PATH = params["data"]["processed_path"]
RANDOM_STATE = params["data"]["random_state"]

DROP_COLS = ["Country", "End_Lat", "End_Lng", "Wind_Chill(F)", "Description", "Turning_Loop"]
DROP_FOR_MODEL = [
    "ID", "Start_Time", "End_Time", "Weather_Timestamp",
    "Street", "Zipcode", "Airport_Code", "City", "County",
    "start_hour", "start_dayofweek", "start_month",
    "is_morning_rush", "is_evening_rush",
    "start_day", "start_minute",
    "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    df.loc[~df["Temperature(F)"].between(-80, 140), "Temperature(F)"] = np.nan
    df.loc[df["Wind_Speed(mph)"] > 200, "Wind_Speed(mph)"] = np.nan
    df.loc[df["Visibility(mi)"] > 20, "Visibility(mi)"] = np.nan
    df.loc[~df["Pressure(in)"].between(25, 35), "Pressure(in)"] = np.nan

    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(np.int8)

    df.dropna(subset=["Start_Time", "End_Time"], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["start_year"] = df["Start_Time"].dt.year
    df["start_month"] = df["Start_Time"].dt.month
    df["start_day"] = df["Start_Time"].dt.day
    df["start_hour"] = df["Start_Time"].dt.hour
    df["start_minute"] = df["Start_Time"].dt.minute
    df["start_dayofweek"] = df["Start_Time"].dt.dayofweek
    df["is_weekend"] = df["start_dayofweek"].isin([5, 6]).astype(np.int8)

    df["is_morning_rush"] = df["start_hour"].between(7, 9).astype(np.int8)
    df["is_evening_rush"] = df["start_hour"].between(16, 18).astype(np.int8)
    df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(np.int8)

    df["duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    df.loc[df["duration_min"] < 0, "duration_min"] = np.nan
    p99 = df["duration_min"].quantile(0.99)
    df = df[df["duration_min"].between(0, p99)].copy()

    df["hour_sin"] = np.sin(2 * np.pi * df["start_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["start_hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["start_dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["start_dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["start_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["start_month"] / 12)
    return df


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df = clean(df)
    df = engineer_features(df)

    drop_cols = [c for c in DROP_FOR_MODEL if c in df.columns]
    X = df.drop(columns=drop_cols + ["Severity"])
    y = df["Severity"]

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    out = X.copy()
    out["Severity"] = y
    out.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed shape: {out.shape}")
    print(f"Saved to: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
    