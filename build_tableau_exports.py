from pathlib import Path
import os

import pandas as pd


DATASET_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / "sobhanmoosavi" / "us-accidents" / "versions" / "13"
OUT_DIR = Path("/Applications/GitHub/US Accidents (2016-2023)/tableau_exports")


USECOLS = [
    "ID",
    "Source",
    "Severity",
    "Start_Time",
    "Start_Lat",
    "Start_Lng",
    "State",
    "County",
    "Timezone",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Direction",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition",
    "Amenity",
    "Crossing",
    "Junction",
    "Sunrise_Sunset",
]


def collapse_categories(series: pd.Series, top_n: int = 8) -> pd.Series:
    series = series.fillna("Missing").astype(str)
    top = series.value_counts().head(top_n).index
    return series.where(series.isin(top), "Other")


def build_master_extract(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=USECOLS, nrows=180_000)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce", format="mixed")
    df = df.dropna(subset=["Start_Time", "Severity", "Start_Lat", "Start_Lng"])

    df["Year"] = df["Start_Time"].dt.year
    df["Month"] = df["Start_Time"].dt.month
    df["Month_Name"] = df["Start_Time"].dt.month_name()
    df["Hour"] = df["Start_Time"].dt.hour
    df["Weekday"] = df["Start_Time"].dt.day_name()
    df["Severe_Flag"] = (df["Severity"] >= 3).astype(int)
    df["Severity_Group"] = df["Severe_Flag"].map({0: "Mild (1-2)", 1: "Severe (3-4)"})
    df["Weather_Group"] = collapse_categories(df["Weather_Condition"], top_n=8)
    df["Wind_Group"] = collapse_categories(df["Wind_Direction"], top_n=8)
    df["Day_Night"] = df["Sunrise_Sunset"].fillna("Unknown")
    return df


def export_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    master_cols = [
        "ID",
        "Source",
        "Severity",
        "Severity_Group",
        "Severe_Flag",
        "Start_Time",
        "Year",
        "Month",
        "Month_Name",
        "Hour",
        "Weekday",
        "State",
        "County",
        "Timezone",
        "Start_Lat",
        "Start_Lng",
        "Temperature(F)",
        "Wind_Chill(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Precipitation(in)",
        "Weather_Condition",
        "Weather_Group",
        "Wind_Direction",
        "Wind_Group",
        "Amenity",
        "Crossing",
        "Junction",
        "Day_Night",
    ]
    df[master_cols].to_csv(out_dir / "01_master_sample.csv", index=False)

    (
        df.groupby("Hour")
        .size()
        .reset_index(name="Accident_Count")
        .to_csv(out_dir / "02_accidents_by_hour.csv", index=False)
    )

    (
        df.groupby(["Hour", "Severity_Group"])
        .size()
        .reset_index(name="Accident_Count")
        .to_csv(out_dir / "03_hour_by_severity.csv", index=False)
    )

    (
        df.groupby(["Month", "Month_Name"])
        .size()
        .reset_index(name="Accident_Count")
        .sort_values("Month")
        .to_csv(out_dir / "04_accidents_by_month.csv", index=False)
    )

    (
        df.groupby(["Weather_Group", "Severity_Group"])
        .size()
        .reset_index(name="Accident_Count")
        .to_csv(out_dir / "05_weather_by_severity.csv", index=False)
    )

    (
        df.groupby("State")
        .agg(
            Accident_Count=("ID", "count"),
            Severe_Rate=("Severe_Flag", "mean"),
            Avg_Visibility=("Visibility(mi)", "mean"),
            Avg_Precipitation=("Precipitation(in)", "mean"),
        )
        .reset_index()
        .sort_values("Accident_Count", ascending=False)
        .to_csv(out_dir / "06_state_summary.csv", index=False)
    )

    (
        df[["Visibility(mi)", "Pressure(in)", "Wind_Chill(F)", "Precipitation(in)", "Humidity(%)", "Wind_Speed(mph)"]]
        .corr(numeric_only=True)
        .reset_index()
        .rename(columns={"index": "Feature"})
        .to_csv(out_dir / "07_numeric_correlation.csv", index=False)
    )

    (
        df.groupby(["Day_Night", "Severity_Group"])
        .size()
        .reset_index(name="Accident_Count")
        .to_csv(out_dir / "08_daynight_by_severity.csv", index=False)
    )

    (
        df.groupby(["Source", "Severity_Group"])
        .size()
        .reset_index(name="Accident_Count")
        .to_csv(out_dir / "09_source_by_severity.csv", index=False)
    )

    (
        df[["Start_Lat", "Start_Lng", "Severity_Group"]]
        .sample(min(40000, len(df)), random_state=42)
        .to_csv(out_dir / "10_map_sample.csv", index=False)
    )


def main() -> None:
    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("US Accidents CSV not found in kagglehub cache.")
    df = build_master_extract(csv_files[0])
    export_tables(df, OUT_DIR)
    print(f"Tableau exports created in: {OUT_DIR}")


if __name__ == "__main__":
    main()
