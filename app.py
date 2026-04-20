from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOCAL_DATASET_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / "sobhanmoosavi" / "us-accidents" / "versions" / "13"
CACHE_DIR = Path("/Applications/GitHub/US Accidents (2016-2023)/data_cache")

PALETTE = {
    "ink": "#1F2933",
    "teal": "#0F766E",
    "coral": "#E76F51",
    "gold": "#E9C46A",
    "sky": "#4C78A8",
    "mist": "#F5F7FA",
    "line": "#D7DEE7",
    "success": "#2A9D8F",
}

VARIABLES = [
    {"Feature": "Source", "Type": "Categorical", "Description": "Traffic data provider that reported the event.", "Collection": "Collected from MapQuest Traffic and Microsoft Bing Maps Traffic APIs."},
    {"Feature": "Severity", "Type": "Ordinal", "Description": "Accident severity label from 1 to 4.", "Collection": "Provided by the traffic feed."},
    {"Feature": "Start_Time", "Type": "Datetime", "Description": "Accident start timestamp.", "Collection": "Reported with the event and used to derive hour, month, and weekday."},
    {"Feature": "Street", "Type": "Categorical", "Description": "Street name near the accident location.", "Collection": "Added through reverse geocoding after matching the event coordinates to an address context."},
    {"Feature": "State", "Type": "Categorical", "Description": "State abbreviation for the accident location.", "Collection": "Added through reverse geocoding after matching the event coordinates to an address context."},
    {"Feature": "Start_Lat", "Type": "Float", "Description": "Starting latitude coordinate.", "Collection": "Reported by the traffic provider."},
    {"Feature": "Start_Lng", "Type": "Float", "Description": "Starting longitude coordinate.", "Collection": "Reported by the traffic provider."},
    {"Feature": "Distance(mi)", "Type": "Float", "Description": "Length of road segment affected in miles.", "Collection": "Derived from the mapped accident segment."},
    {"Feature": "Timezone", "Type": "Categorical", "Description": "Timezone of the accident location.", "Collection": "Added from geographic lookup during augmentation."},
    {"Feature": "Wind_Chill(F)", "Type": "Float", "Description": "Perceived temperature in Fahrenheit.", "Collection": "Matched from the nearest airport weather station observation."},
    {"Feature": "Pressure(in)", "Type": "Float", "Description": "Air pressure in inches.", "Collection": "Matched from the nearest airport weather station observation."},
    {"Feature": "Visibility(mi)", "Type": "Float", "Description": "Visibility in miles.", "Collection": "Matched from the nearest airport weather station observation."},
    {"Feature": "Wind_Direction", "Type": "Categorical", "Description": "Wind direction label such as N, SW, Calm, or Variable.", "Collection": "Matched from the nearest airport weather station observation."},
    {"Feature": "Precipitation(in)", "Type": "Float", "Description": "Observed precipitation amount in inches.", "Collection": "Matched from the nearest airport weather station observation."},
    {"Feature": "Weather_Condition", "Type": "Categorical", "Description": "Text description of the weather, such as Rain, Fog, or Snow.", "Collection": "Matched from the nearest airport weather station observation."},
    {"Feature": "Amenity", "Type": "Boolean", "Description": "Whether an amenity exists near the accident.", "Collection": "Added from OpenStreetMap point-of-interest augmentation."},
    {"Feature": "Crossing", "Type": "Boolean", "Description": "Whether a crossing is near the accident.", "Collection": "Added from OpenStreetMap point-of-interest augmentation."},
    {"Feature": "Junction", "Type": "Boolean", "Description": "Whether a junction is near the accident.", "Collection": "Added from OpenStreetMap point-of-interest augmentation."},
]

LEAKAGE_EXCLUSIONS = {
    "Severity": "Target variable.",
    "Distance(mi)": "Often reflects event extent after the accident is already underway.",
}

HYPOTHESIS_EXCLUSIONS = {
    "Severity": "Target variable.",
    "Distance(mi)": "Too close to accident extent / impact rather than a pre-accident driver.",
}

MODEL_FEATURES = [
    "Source",
    "Start_Lat",
    "Start_Lng",
    "Timezone",
    "Wind_Chill(F)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Direction",
    "Precipitation(in)",
    "Weather_Condition",
    "Amenity",
    "Crossing",
    "Junction",
    "Hour",
    "Month",
    "Weekday",
]


def find_csv_path() -> Path:
    csv_files = sorted(LOCAL_DATASET_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("US Accidents CSV not found in the local kagglehub cache.")
    return csv_files[0]


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=PALETTE["ink"], size=13),
        margin=dict(l=20, r=20, t=60, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["line"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["line"], zeroline=False)
    return fig


def format_num(value) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return str(value)


def collapse_categories(series: pd.Series, top_n: int = 8) -> pd.Series:
    series = series.fillna("Missing").astype(str)
    top = series.value_counts().head(top_n).index
    return series.where(series.isin(top), "Other")


def p_value_text(p_value: float) -> str:
    if p_value < 0.001:
        return "< 0.001"
    return f"{p_value:.4f}"


def two_proportion_z_test(success_a, size_a, success_b, size_b):
    p_pool = (success_a + success_b) / (size_a + size_b)
    se = np.sqrt(p_pool * (1 - p_pool) * ((1 / size_a) + (1 / size_b)))
    if se == 0:
        return np.nan, np.nan
    z_stat = ((success_a / size_a) - (success_b / size_b)) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value


def cramers_v(contingency: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = contingency.shape
    denom = min(k - 1, r - 1)
    return np.nan if denom <= 0 else np.sqrt((chi2 / n) / denom)


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    n1, n2 = len(group_a), len(group_b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = group_a.std(ddof=1), group_b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return 0.0 if pooled == 0 else (group_a.mean() - group_b.mean()) / pooled


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meta in VARIABLES:
        feature = meta["Feature"]
        series = df[feature]
        if meta["Type"] in {"Float", "Ordinal"}:
            rng = f"{format_num(pd.to_numeric(series, errors='coerce').min())} to {format_num(pd.to_numeric(series, errors='coerce').max())}"
        elif meta["Type"] == "Datetime":
            valid = series.dropna()
            rng = f"{valid.min().strftime('%Y-%m-%d %H:%M')} to {valid.max().strftime('%Y-%m-%d %H:%M')}" if not valid.empty else "N/A"
        else:
            vals = ", ".join(series.dropna().astype(str).value_counts().head(5).index.tolist())
            rng = vals if vals else "N/A"
        rows.append({
            "Feature": feature,
            "Type": meta["Type"],
            "Description": meta["Description"],
            "How Collected": meta["Collection"],
            "Observed Range / Top Values": rng,
        })
    return pd.DataFrame(rows)


def load_profile_sample(csv_path: Path) -> pd.DataFrame:
    usecols = [meta["Feature"] for meta in VARIABLES]
    df = pd.read_csv(csv_path, usecols=usecols, nrows=120_000)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce", format="mixed")
    df["Hour"] = df["Start_Time"].dt.hour
    df["Month"] = df["Start_Time"].dt.month
    df["Weekday"] = df["Start_Time"].dt.day_name()
    df["Severe"] = df["Severity"] >= 3
    return df


def build_overview_metrics(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        ("Rows used in app sample", format_num(len(df))),
        ("Coverage window", f"{df['Start_Time'].min().strftime('%Y-%m-%d')} to {df['Start_Time'].max().strftime('%Y-%m-%d')}"),
        ("Severe accident share", f"{df['Severe'].mean():.2%}"),
        ("Weather fields with missingness", format_num(df[['Wind_Chill(F)', 'Pressure(in)', 'Visibility(mi)', 'Precipitation(in)', 'Weather_Condition']].isna().any(axis=1).sum())),
    ]


def build_road_danger_data(csv_path: Path) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "road_danger_full.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    usecols = ["Street", "State", "Start_Lat", "Start_Lng", "Severity"]
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=300_000):
        chunk = chunk.dropna(subset=["Street", "State", "Start_Lat", "Start_Lng", "Severity"]).copy()
        chunk["Street"] = chunk["Street"].astype(str).str.strip()
        chunk["State"] = chunk["State"].astype(str).str.strip()
        chunk = chunk[(chunk["Street"] != "") & (chunk["State"] != "")]
        chunk["Road_Label"] = chunk["Street"] + ", " + chunk["State"]
        chunk["Severe_Flag"] = (pd.to_numeric(chunk["Severity"], errors="coerce") >= 3).astype(int)
        grouped = (
            chunk.groupby(["Road_Label", "Street", "State"], as_index=False)
            .agg(
                lat_sum=("Start_Lat", "sum"),
                lng_sum=("Start_Lng", "sum"),
                Accident_Count=("Severity", "size"),
                severe_sum=("Severe_Flag", "sum"),
                severity_sum=("Severity", "sum"),
            )
        )
        chunks.append(grouped)

    combined = pd.concat(chunks, ignore_index=True)
    final = (
        combined.groupby(["Road_Label", "Street", "State"], as_index=False)
        .agg(
            lat_sum=("lat_sum", "sum"),
            lng_sum=("lng_sum", "sum"),
            Accident_Count=("Accident_Count", "sum"),
            severe_sum=("severe_sum", "sum"),
            severity_sum=("severity_sum", "sum"),
        )
    )
    final["Start_Lat"] = final["lat_sum"] / final["Accident_Count"]
    final["Start_Lng"] = final["lng_sum"] / final["Accident_Count"]
    final["Severe_Rate"] = final["severe_sum"] / final["Accident_Count"]
    final["Avg_Severity"] = final["severity_sum"] / final["Accident_Count"]
    final["Weighted_Danger"] = final["Severe_Rate"] * np.log1p(final["Accident_Count"])
    final = final.drop(columns=["lat_sum", "lng_sum", "severe_sum", "severity_sum"])
    final.to_csv(cache_path, index=False)
    return final


def build_road_danger_map(
    df: pd.DataFrame,
    metric: str = "Weighted_Danger",
    selected_states=None,
    min_count: int = 1,
):
    plot_df = df.copy()
    if selected_states:
        plot_df = plot_df[plot_df["State"].isin(selected_states)]
    plot_df = plot_df[plot_df["Accident_Count"] >= min_count].copy()
    plot_df = plot_df.sort_values(metric, ascending=False)
    metric_labels = {
        "Weighted_Danger": "Weighted danger",
        "Severe_Rate": "Severe rate",
        "Accident_Count": "Accident count",
        "Avg_Severity": "Average severity",
    }
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No roads match the current filters.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=18, color=PALETTE["ink"]),
        )
        return style_figure(fig)

    fig = px.scatter_mapbox(
        plot_df,
        lat="Start_Lat",
        lon="Start_Lng",
        size="Accident_Count",
        color=metric,
        hover_name="Road_Label",
        hover_data={
            "Street": False,
            "State": True,
            "Accident_Count": ":,.0f",
            "Severe_Rate": ":.2%",
            "Avg_Severity": ":.2f",
            metric: ":.3f",
            "Start_Lat": False,
            "Start_Lng": False,
        },
        color_continuous_scale=[PALETTE["mist"], PALETTE["gold"], PALETTE["coral"]],
        title=f"Roads that look most dangerous by {metric_labels.get(metric, metric)}",
        zoom=3 if not selected_states else 4.2,
        height=650,
        mapbox_style="carto-positron",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title=metric_labels.get(metric, metric)),
        mapbox=dict(center=dict(lat=float(plot_df["Start_Lat"].mean()), lon=float(plot_df["Start_Lng"].mean()))),
    )
    return style_figure(fig)


def filter_road_danger_data(df: pd.DataFrame, selected_states=None, min_count: int = 1) -> pd.DataFrame:
    filtered = df.copy()
    if selected_states:
        filtered = filtered[filtered["State"].isin(selected_states)]
    filtered = filtered[filtered["Accident_Count"] >= min_count].copy()
    return filtered


def build_road_summary_strip(df: pd.DataFrame, metric: str):
    if df.empty:
        items = [
            ("Roads shown", "0"),
            ("Accidents covered", "0"),
            ("Average severe rate", "0.00%"),
            ("Top road", "None"),
        ]
    else:
        top_row = df.sort_values(metric, ascending=False).iloc[0]
        items = [
            ("Roads shown", format_num(len(df))),
            ("Accidents covered", format_num(int(df["Accident_Count"].sum()))),
            ("Average severe rate", f"{df['Severe_Rate'].mean():.2%}"),
            ("Top road", top_row["Road_Label"]),
        ]
    return html.Div(
        [
            html.Div(
                [html.Span(title, className="summary-label"), html.Span(value, className="summary-value")],
                className="summary-item",
            )
            for title, value in items
        ],
        className="summary-strip compact-strip",
    )


def build_road_top_table(df: pd.DataFrame, metric: str):
    if df.empty:
        return html.Div("No roads match the current filters.", className="text-muted")
    cols = ["Road_Label", "State", "Accident_Count", "Severe_Rate", "Avg_Severity", metric]
    table_df = df.sort_values(metric, ascending=False).head(10)[cols].copy()
    table_df["Severe_Rate"] = table_df["Severe_Rate"].map(lambda x: f"{x:.2%}")
    table_df["Avg_Severity"] = table_df["Avg_Severity"].map(lambda x: f"{x:.2f}")
    table_df[metric] = table_df[metric].map(lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else x)
    label = {
        "Weighted_Danger": "Weighted danger",
        "Severe_Rate": "Severe rate",
        "Accident_Count": "Accident count",
        "Avg_Severity": "Average severity",
    }.get(metric, metric)
    return html.Div(
        [
            html.H5("Top roads under current filters", className="section-title mb-2"),
            dash_table.DataTable(
                data=table_df.to_dict("records"),
                columns=[
                    {"name": "Road", "id": "Road_Label"},
                    {"name": "State", "id": "State"},
                    {"name": "Accidents", "id": "Accident_Count"},
                    {"name": "Severe rate", "id": "Severe_Rate"},
                    {"name": "Avg severity", "id": "Avg_Severity"},
                    {"name": label, "id": metric},
                ],
                style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "padding": "8px", "fontSize": "13px"},
                style_header={"backgroundColor": PALETTE["ink"], "color": "white", "fontWeight": "bold"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": PALETTE["mist"]}],
            ),
        ]
    )


def build_overview_figures(df: pd.DataFrame):
    severity_counts = df["Severity"].value_counts().sort_index().reset_index()
    severity_counts.columns = ["Severity", "Count"]
    fig_severity = px.bar(
        severity_counts,
        x="Severity",
        y="Count",
        text_auto=True,
        color_discrete_sequence=[PALETTE["coral"]],
        title="Severity distribution",
    )

    hourly = df.groupby("Hour").size().reset_index(name="Count")
    fig_hour = px.line(
        hourly,
        x="Hour",
        y="Count",
        markers=True,
        title="Accident frequency by hour",
    )
    fig_hour.update_traces(line_color=PALETTE["teal"], marker_color=PALETTE["gold"])

    weather = df.assign(Weather_Group=collapse_categories(df["Weather_Condition"], top_n=6))
    weather_sev = (
        weather.groupby(["Weather_Group", "Severe"])
        .size()
        .reset_index(name="Count")
    )
    weather_sev["Severity Group"] = weather_sev["Severe"].map({False: "Mild (1-2)", True: "Severe (3-4)"})
    fig_weather = px.bar(
        weather_sev,
        x="Weather_Group",
        y="Count",
        color="Severity Group",
        barmode="group",
        title="Weather condition vs severity split",
        color_discrete_sequence=[PALETTE["sky"], PALETTE["coral"]],
    )

    coord_df = df.dropna(subset=["Start_Lat", "Start_Lng"]).copy()
    fig_geo = px.density_heatmap(
        coord_df,
        x="Start_Lng",
        y="Start_Lat",
        nbinsx=70,
        nbinsy=42,
        title="US accident density in latitude / longitude space",
        color_continuous_scale=[PALETTE["mist"], PALETTE["gold"], PALETTE["coral"]],
    )
    fig_geo.update_traces(
        hovertemplate="Longitude bin: %{x}<br>Latitude bin: %{y}<br>Accidents: %{z}<extra></extra>"
    )

    missing = (
        df[["Start_Time", "Source", "Start_Lat", "Start_Lng", "Wind_Chill(F)", "Pressure(in)", "Visibility(mi)", "Precipitation(in)", "Weather_Condition"]]
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    missing.columns = ["Feature", "Missing %"]
    fig_missing = px.bar(
        missing,
        x="Missing %",
        y="Feature",
        orientation="h",
        title="Missingness check",
        color_discrete_sequence=["#C8D2DC"],
    )

    return [style_figure(fig) for fig in [fig_severity, fig_hour, fig_weather, fig_geo, fig_missing]]


def test_feature(df: pd.DataFrame, feature: str, feature_type: str):
    if feature in HYPOTHESIS_EXCLUSIONS:
        return None

    if feature_type in {"Float", "Ordinal"}:
        subset = df[[feature, "Severe"]].copy()
        subset[feature] = pd.to_numeric(subset[feature], errors="coerce")
        subset = subset.dropna()
        a = subset.loc[subset["Severe"], feature]
        b = subset.loc[~subset["Severe"], feature]
        if len(a) < 2 or len(b) < 2:
            return None
        stat, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return {"Feature": feature, "Type": feature_type, "Test": "Welch t-test", "Statistic": stat, "P-Value": p_value, "Effect Size": cohens_d(a, b), "Details": f"mean severe={a.mean():.2f}, mild={b.mean():.2f}"}

    if feature_type == "Boolean":
        subset = df[[feature, "Severe"]].dropna()
        yes = subset[subset[feature].astype(bool)]
        no = subset[~subset[feature].astype(bool)]
        if len(yes) < 2 or len(no) < 2:
            return None
        stat, p_value = two_proportion_z_test(yes["Severe"].sum(), len(yes), no["Severe"].sum(), len(no))
        return {"Feature": feature, "Type": feature_type, "Test": "Two-proportion z-test", "Statistic": stat, "P-Value": p_value, "Effect Size": yes["Severe"].mean() - no["Severe"].mean(), "Details": f"severe rate true={yes['Severe'].mean():.2%}, false={no['Severe'].mean():.2%}"}

    if feature_type == "Datetime":
        subset = df[[feature, "Severe"]].dropna().copy()
        subset["Time Bin"] = pd.cut(subset[feature].dt.hour, bins=[-1, 5, 11, 17, 23], labels=["Night", "Morning", "Afternoon", "Evening"])
        contingency = pd.crosstab(subset["Time Bin"], subset["Severe"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None
        stat, p_value, _, _ = stats.chi2_contingency(contingency)
        return {"Feature": feature, "Type": feature_type, "Test": "Chi-square on time bins", "Statistic": stat, "P-Value": p_value, "Effect Size": cramers_v(contingency), "Details": "time binned into night / morning / afternoon / evening"}

    subset = df[[feature, "Severe"]].copy()
    subset[feature] = collapse_categories(subset[feature], top_n=8)
    contingency = pd.crosstab(subset[feature], subset["Severe"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    stat, p_value, _, _ = stats.chi2_contingency(contingency)
    top_label = subset[feature].value_counts().idxmax()
    return {"Feature": feature, "Type": feature_type, "Test": "Chi-square test", "Statistic": stat, "P-Value": p_value, "Effect Size": cramers_v(contingency), "Details": f"top grouped label={top_label}; rare labels collapsed into Other"}


def compute_hypothesis_results(df: pd.DataFrame):
    rows = []
    excluded = [{"Feature": key, "Reason Excluded": value} for key, value in HYPOTHESIS_EXCLUSIONS.items()]
    for meta in VARIABLES:
        result = test_feature(df, meta["Feature"], meta["Type"])
        if result is not None:
            rows.append(result)
    tests = pd.DataFrame(rows)
    tests["P-Value Label"] = tests["P-Value"].apply(p_value_text)
    tests["Significant (0.05)"] = np.where(tests["P-Value"] < 0.05, "Yes", "No")
    tests["Abs Effect"] = pd.to_numeric(tests["Effect Size"], errors="coerce").abs()
    tests = tests.sort_values(["P-Value", "Abs Effect"], ascending=[True, False]).reset_index(drop=True)

    pca_cols = ["Start_Lat", "Start_Lng", "Wind_Chill(F)", "Pressure(in)", "Visibility(mi)", "Precipitation(in)", "Hour", "Month"]
    pca_df = df[pca_cols].apply(pd.to_numeric, errors="coerce").dropna().head(25000)
    pca = PCA(n_components=3, random_state=42)
    scaled = StandardScaler().fit_transform(pca_df)
    pca.fit(scaled)
    explained = pd.DataFrame({"Component": ["PC1", "PC2", "PC3"], "Explained Variance Ratio": pca.explained_variance_ratio_})

    return {"tests_df": tests, "excluded_df": pd.DataFrame(excluded), "pca_explained": explained}


def compute_model_results(df: pd.DataFrame):
    model_df = df[["Severity"] + MODEL_FEATURES].copy()
    model_df["Target"] = (model_df["Severity"] >= 3).astype(int)
    model_df = model_df.drop(columns=["Severity"])

    categorical = ["Source", "Timezone", "Wind_Direction", "Weather_Condition", "Weekday"]
    numeric = ["Start_Lat", "Start_Lng", "Wind_Chill(F)", "Pressure(in)", "Visibility(mi)", "Precipitation(in)", "Hour", "Month"]
    boolean = ["Amenity", "Crossing", "Junction"]

    X = model_df[categorical + numeric + boolean]
    y = model_df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pre = ColumnTransformer(
        [
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("bool", "passthrough", boolean),
        ]
    )
    model = Pipeline(
        [
            ("pre", pre),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC AUC": roc_auc_score(y_test, prob),
    }

    cm = confusion_matrix(y_test, pred)
    cm_df = pd.DataFrame(cm, index=["Actual Mild", "Actual Severe"], columns=["Pred Mild", "Pred Severe"])

    feature_names = model.named_steps["pre"].get_feature_names_out()
    importances = pd.DataFrame({"Feature": feature_names, "Importance": model.named_steps["rf"].feature_importances_})
    importances["Raw Feature"] = (
        importances["Feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
        .str.replace("bool__", "", regex=False)
        .str.split("_")
        .str[0]
    )
    grouped = importances.groupby("Raw Feature", as_index=False)["Importance"].sum().sort_values("Importance", ascending=False).head(10)

    return {"metrics": metrics, "confusion": cm_df, "importances": grouped}


def build_hypothesis_figures(results):
    top = results["tests_df"].head(12).copy()
    top["Score"] = -np.log10(top["P-Value"].clip(lower=1e-300))
    top["Label"] = top["Feature"] + " | " + top["Test"]
    fig_rank = px.bar(
        top.sort_values("Score"),
        x="Score",
        y="Label",
        orientation="h",
        color="Type",
        title="Screened variables ranked by statistical signal",
        color_discrete_sequence=[PALETTE["teal"], PALETTE["coral"], PALETTE["gold"], PALETTE["sky"]],
    )

    type_counts = results["tests_df"].groupby(["Type", "Significant (0.05)"]).size().reset_index(name="Count")
    fig_counts = px.bar(
        type_counts,
        x="Type",
        y="Count",
        color="Significant (0.05)",
        barmode="group",
        title="How many screened variables stayed significant",
        color_discrete_sequence=[PALETTE["sky"], PALETTE["coral"]],
    )

    fig_pca = px.bar(
        results["pca_explained"],
        x="Component",
        y="Explained Variance Ratio",
        text_auto=".2%",
        title="PCA explained variance on the clean numeric set",
        color="Component",
        color_discrete_sequence=[PALETTE["teal"], PALETTE["gold"], PALETTE["coral"]],
    )
    fig_pca.update_yaxes(tickformat=".0%")
    return [style_figure(fig) for fig in [fig_rank, fig_counts, fig_pca]]


def build_model_figures(model_results):
    metric_df = pd.DataFrame({"Metric": list(model_results["metrics"].keys()), "Value": list(model_results["metrics"].values())})
    fig_metrics = px.bar(metric_df, x="Metric", y="Value", text_auto=".3f", title="Random forest baseline metrics", color="Metric", color_discrete_sequence=[PALETTE["teal"], PALETTE["coral"], PALETTE["gold"], PALETTE["sky"], PALETTE["success"]])
    fig_metrics.update_yaxes(range=[0, 1], tickformat=".0%")

    fig_cm = px.imshow(
        model_results["confusion"],
        text_auto=True,
        color_continuous_scale=[[0, PALETTE["mist"]], [1, PALETTE["coral"]]],
        title="Confusion matrix on the holdout set",
    )

    fig_imp = px.bar(
        model_results["importances"].sort_values("Importance"),
        x="Importance",
        y="Raw Feature",
        orientation="h",
        title="Most influential raw feature groups in the baseline model",
        color_discrete_sequence=[PALETTE["gold"]],
    )
    return [style_figure(fig) for fig in [fig_metrics, fig_cm, fig_imp]]


def metric_card(title: str, value: str, delay: int = 0):
    return dbc.Card(
        dbc.CardBody([html.Div(title, className="eyebrow"), html.H3(value, className="metric-value mb-0")]),
        className=f"metric-card fade-up delay-{delay}",
    )


CSV_PATH = find_csv_path()
PROFILE_DF = load_profile_sample(CSV_PATH)
SUMMARY_DF = build_summary_table(PROFILE_DF)
OVERVIEW_METRICS = build_overview_metrics(PROFILE_DF)
OVERVIEW_FIGS = build_overview_figures(PROFILE_DF)
ROAD_DANGER_DF = build_road_danger_data(CSV_PATH)
STATE_OPTIONS = sorted(ROAD_DANGER_DF["State"].dropna().unique().tolist())
HYPOTHESIS_RESULTS = compute_hypothesis_results(PROFILE_DF)
HYPOTHESIS_FIGS = build_hypothesis_figures(HYPOTHESIS_RESULTS)
MODEL_RESULTS = compute_model_results(PROFILE_DF)
MODEL_FIGS = build_model_figures(MODEL_RESULTS)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
app.title = "Road Warning Intelligence"


def nav_bar():
    return dbc.Navbar(
        dbc.Container(
            [
                html.Div(
                    [
                        html.Div("Road Warning Intelligence", className="brand-title"),
                        html.Div("a navigation safety layer for high-impact roads", className="brand-subtitle"),
                    ]
                ),
                dbc.Nav(
                    [
                        dbc.NavLink("Overview", href="/", className="nav-link-custom"),
                        dbc.NavLink("Hypotheses", href="/hypotheses", className="nav-link-custom"),
                    ],
                    pills=False,
                    className="ms-auto",
                ),
            ]
        ),
        className="top-nav",
        dark=False,
    )


def overview_page():
    fig_severity, fig_hour, fig_weather, fig_geo, fig_missing = OVERVIEW_FIGS
    road_map = build_road_danger_map(ROAD_DANGER_DF)
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H1("A Warning Layer For High-Risk Roads", className="hero-title fade-up"),
                            html.P(
                                "Built to identify risky road corridors, surface warning signals before a driver enters them, and support municipalities, federal agencies, and navigation platforms with a cleaner safety layer.",
                                className="hero-copy fade-up delay-1",
                            ),
                        ],
                        className="hero-block",
                    ),
                    width=12,
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(title, className="summary-label"),
                                    html.Span(value, className="summary-value"),
                                ],
                                className="summary-item",
                            )
                            for title, value in OVERVIEW_METRICS
                        ],
                        className="summary-strip fade-up",
                    ),
                    width=12,
                ),
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Where The Data Comes From", className="section-title"),
                                    html.Ul(
                                        [
                                            html.Li("Traffic events came from MapQuest Traffic and Microsoft Bing Maps Traffic APIs."),
                                            html.Li("Reverse geocoding added location context such as timezone and address-level fields."),
                                            html.Li("Weather was matched from nearby airport weather stations."),
                                            html.Li("OpenStreetMap added road-context flags like Amenity, Crossing, and Junction."),
                                        ],
                                        className="mb-0",
                                    ),
                                ]
                            ),
                            className="panel-card h-100 fade-up delay-1",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Why this view exists", className="section-title"),
                                    html.P("Everything on this page supports one product question: where should a warning system intervene first, and under what conditions does the road risk rise?", className="text-muted mb-2"),
                                    html.P("That is why the page stays focused on time rhythm, weather context, road concentration, and data quality.", className="text-muted mb-0"),
                                ]
                            ),
                            className="panel-card h-100",
                        ),
                        md=6,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Graph(figure=fig_hour, config={"displayModeBar": False}),
                                    html.P(
                                        "This is the anchor chart. Commute-hour clustering is one of the clearest signals that a warning system should not treat risk as static across the day.",
                                        className="text-muted mt-2 mb-0",
                                    ),
                                ]
                            ),
                            className="panel-card",
                        ),
                        md=12,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Severity balance", className="section-title"),
                                    html.P(
                                        f"The sample is not wildly imbalanced: severe crashes make up {PROFILE_DF['Severe'].mean():.2%} of the records. That is strong enough to matter, but not so lopsided that every later chart has to fight class skew first.",
                                        className="text-muted mb-3",
                                    ),
                                    dcc.Graph(figure=fig_weather, config={"displayModeBar": False}),
                                ]
                            ),
                            className="panel-card",
                        ),
                        md=12,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Road warning map", className="section-title"),
                                    html.P(
                                        "This view aggregates accident history by road name and state, then ranks roads by a switchable risk metric. It is designed as a warning and prioritization layer: which roads deserve alerts first, where agencies should focus, and which corridors a navigation partner could flag to drivers.",
                                        className="text-muted mb-2",
                                    ),
                                    html.Div(id="road-risk-summary", children=build_road_summary_strip(filter_road_danger_data(ROAD_DANGER_DF, min_count=10), "Weighted_Danger"), className="mb-3"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("States", className="eyebrow"),
                                                    dcc.Dropdown(
                                                        id="road-risk-states",
                                                        options=[{"label": state, "value": state} for state in STATE_OPTIONS],
                                                        value=[],
                                                        multi=True,
                                                        placeholder="All states",
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Danger metric", className="eyebrow"),
                                                    dcc.Dropdown(
                                                        id="road-risk-metric",
                                                        options=[
                                                            {"label": "Weighted danger", "value": "Weighted_Danger"},
                                                            {"label": "Severe rate", "value": "Severe_Rate"},
                                                            {"label": "Accident count", "value": "Accident_Count"},
                                                            {"label": "Average severity", "value": "Avg_Severity"},
                                                        ],
                                                        value="Weighted_Danger",
                                                        clearable=False,
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Minimum accidents", className="eyebrow"),
                                                    dcc.Slider(
                                                        id="road-risk-min-count",
                                                        min=1,
                                                        max=50,
                                                        step=1,
                                                        value=10,
                                                        marks={1: "1", 10: "10", 25: "25", 50: "50"},
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                        ],
                                        className="g-2 mb-3 align-items-end",
                                    ),
                                    html.Div(id="road-risk-top-table", className="mb-3"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button("Download Interactive HTML", id="download-road-html-btn", color="dark", className="w-100"),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Button("Download Filtered CSV", id="download-road-csv-btn", outline=True, color="dark", className="w-100"),
                                                md=4,
                                            ),
                                        ],
                                        className="g-3 mb-3",
                                    ),
                                    dcc.Graph(
                                        id="road-risk-map",
                                        figure=road_map,
                                        config={
                                            "displayModeBar": True,
                                            "scrollZoom": True,
                                            "displaylogo": False,
                                            "modeBarButtonsToRemove": [
                                                "select2d",
                                                "lasso2d",
                                                "zoom2d",
                                                "pan2d",
                                                "zoomIn2d",
                                                "zoomOut2d",
                                                "autoScale2d",
                                                "resetScale2d",
                                            ],
                                        },
                                    ),
                                    dcc.Download(id="download-road-html"),
                                    dcc.Download(id="download-road-csv"),
                                    
                                ]
                            ),
                            className="panel-card",
                        ),
                        md=12,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Coordinate density", className="section-title"),
                                    html.P(
                                        "A hexbin-style density view of US accident coordinates. This is the spatial evidence layer, without the distraction of map tiles or decorative geography.",
                                        className="text-muted mb-3",
                                    ),
                                    dcc.Graph(figure=fig_geo, config={"displayModeBar": False}),
                                ]
                            ),
                            className="panel-card",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Cleaning checkpoint", className="section-title"),
                                    dcc.Graph(figure=fig_missing, config={"displayModeBar": False}),
                                    html.P("Keep fields with tolerable missingness and a coherent collection story.", className="text-muted mt-2 mb-0"),
                                ]
                            ),
                            className="panel-card",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Field Guide", className="section-title"),
                                    html.P(
                                        "Reference only.",
                                        className="text-muted mb-3",
                                    ),
                                    html.Details(
                                        [
                                            html.Summary("Open variable reference", className="details-summary"),
                                            html.Div(
                                                dash_table.DataTable(
                                                    data=SUMMARY_DF.to_dict("records"),
                                                    columns=[{"name": c, "id": c} for c in SUMMARY_DF.columns],
                                                    page_size=8,
                                                    sort_action="native",
                                                    style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "padding": "10px", "fontSize": "13px"},
                                                    style_header={"backgroundColor": PALETTE["ink"], "color": "white", "fontWeight": "bold"},
                                                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": PALETTE["mist"]}],
                                                ),
                                                className="details-body",
                                            ),
                                        ],
                                        className="details-block",
                                    ),
                                ]
                            ),
                            className="panel-card",
                        ),
                        md=7,
                    ),
                ],
                className="g-3 mb-4",
            ),
        ],
        fluid=True,
        className="page-shell",
    )


def hypotheses_page():
    fig_rank, fig_counts, fig_pca = HYPOTHESIS_FIGS
    fig_metrics, fig_cm, fig_imp = MODEL_FIGS
    tests_df = HYPOTHESIS_RESULTS["tests_df"][["Feature", "Type", "Test", "P-Value Label", "Effect Size", "Significant (0.05)", "Details"]].copy()
    tests_df["Effect Size"] = tests_df["Effect Size"].apply(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")
    top_table_df = tests_df.head(8).copy()
    excluded_df = HYPOTHESIS_RESULTS["excluded_df"]
    top_five = HYPOTHESIS_RESULTS["tests_df"].head(5)
    leakage_df = pd.DataFrame(
        [
            {"Rule": "Drop post-event fields", "Why": "Anything observed after the accident starts can leak outcome information.", "Applied To": "Distance(mi) and any end-state style fields."},
            {"Rule": "Avoid duplicate time signals", "Why": "A matched weather timestamp is not a fresh cause variable if Start_Time already exists.", "Applied To": "Weather_Timestamp excluded; Start_Time kept as the canonical clock."},
            {"Rule": "Do not lean on identifier-like location fields", "Why": "Street / zip / county can memorize places instead of explaining general patterns.", "Applied To": "Street, County, Zipcode excluded."},
            {"Rule": "Model only pre-accident context", "Why": "The baseline model should be realistic for later prediction work.", "Applied To": ", ".join(MODEL_FEATURES)},
        ]
    )
    model_metrics = [(name, f"{value:.3f}") for name, value in MODEL_RESULTS["metrics"].items()]

    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H1("What Holds Up When The Product Claim Is Tested", className="hero-title fade-up"),
                            html.P(
                                "The goal is not just to map dangerous roads. It is to defend a warning product: test which signals are real, remove leakage, and show what a disciplined baseline model could power in a routing or alert system.",
                                className="hero-copy fade-up delay-1",
                            ),
                        ],
                        className="hero-block",
                    ),
                    width=12,
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Strongest Signals", className="section-title"),
                                    html.P("Evidence first, then exclusions, then the baseline model.", className="text-muted mb-3"),
                                    html.Ul([html.Li(f"{row['Feature']} via {row['Test']} with p {row['P-Value Label']}") for _, row in top_five.iterrows()], className="mb-3"),
                                    html.P("Weather_Condition stayed significant while Source did not. Good. That means the screen is reacting to context rather than provider trivia.", className="text-muted mb-0"),
                                ]
                            ),
                            className="panel-card h-100 fade-up delay-1",
                        ),
                        md=12,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_rank, config={"displayModeBar": False})), className="panel-card"), md=8), dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_counts, config={"displayModeBar": False})), className="panel-card"), md=4)], className="g-3 mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Significance Screen", className="section-title"),
                                    dash_table.DataTable(
                                        data=top_table_df.to_dict("records"),
                                        columns=[{"name": c, "id": c} for c in tests_df.columns],
                                        page_size=8,
                                        sort_action="native",
                                        style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "padding": "8px", "fontSize": "13px"},
                                        style_header={"backgroundColor": PALETTE["ink"], "color": "white", "fontWeight": "bold"},
                                        style_data_conditional=[{"if": {"filter_query": "{Significant (0.05)} = Yes"}, "backgroundColor": "#E9F7F2"}],
                                    ),
                                    html.Details(
                                        [
                                            html.Summary("Open full test matrix", className="details-summary"),
                                            html.Div(
                                                dash_table.DataTable(
                                                    data=tests_df.to_dict("records"),
                                                    columns=[{"name": c, "id": c} for c in tests_df.columns],
                                                    page_size=10,
                                                    sort_action="native",
                                                    style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "padding": "8px", "fontSize": "13px"},
                                                    style_header={"backgroundColor": PALETTE["ink"], "color": "white", "fontWeight": "bold"},
                                                    style_data_conditional=[{"if": {"filter_query": "{Significant (0.05)} = Yes"}, "backgroundColor": "#E9F7F2"}],
                                                ),
                                                className="details-body",
                                            ),
                                        ],
                                        className="details-block mt-3",
                                    ),
                                ]
                            ),
                            className="panel-card",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("What Stayed Out", className="section-title"),
                                    dash_table.DataTable(
                                        data=excluded_df.to_dict("records"),
                                        columns=[{"name": c, "id": c} for c in excluded_df.columns],
                                        style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "padding": "8px", "fontSize": "13px"},
                                        style_header={"backgroundColor": PALETTE["ink"], "color": "white", "fontWeight": "bold"},
                                    ),
                                ]
                            ),
                            className="panel-card h-100",
                        ),
                        md=4,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Why This Baseline", className="section-title"),
                                    html.P("The first model is a random forest baseline, not because it is the only good choice, but because it can absorb nonlinear interactions among weather, time, and road-context features while still letting us inspect grouped importances.", className="text-muted"),
                                    html.P("It is trained only on pre-accident or immediately available contextual fields: provider, location, weather, POI flags, and derived time-of-day features.", className="text-muted mb-0"),
                                ]
                            ),
                            className="panel-card h-100",
                        ),
                        md=8,
                    ),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_pca, config={"displayModeBar": False})), className="panel-card"), md=4),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(title, className="summary-label"),
                                    html.Span(value, className="summary-value"),
                                ],
                                className="summary-item",
                            )
                            for title, value in model_metrics
                        ],
                        className="summary-strip fade-up mb-3",
                    ),
                    width=12,
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Graph(figure=fig_imp, config={"displayModeBar": False}),
                                    html.P(
                                        "Grouped importances matter more here than another metric recap because they show what the baseline actually leans on.",
                                        className="text-muted mt-2 mb-0",
                                    ),
                                ]
                            ),
                            className="panel-card",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Graph(figure=fig_cm, config={"displayModeBar": False}),
                                ]
                            ),
                            className="panel-card h-100",
                        ),
                        md=5,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(figure=fig_metrics, config={"displayModeBar": False}),
                            ]
                        ),
                        className="panel-card",
                    ),
                    width=12,
                ),
                className="g-3 mb-4",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Leakage Defense", className="section-title"),
                                dash_table.DataTable(
                                    data=leakage_df.to_dict("records"),
                                    columns=[{"name": c, "id": c} for c in leakage_df.columns],
                                    style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "padding": "10px", "fontSize": "13px"},
                                    style_header={"backgroundColor": PALETTE["ink"], "color": "white", "fontWeight": "bold"},
                                ),
                                html.P("Current baseline result on the sample: Accuracy 0.693, F1 0.664, ROC AUC 0.766. That is decent enough to justify continuing, but not so good that it smells like leakage. Which is exactly what we want.", className="text-muted mt-3 mb-0"),
                            ]
                        ),
                        className="panel-card",
                    ),
                    width=12,
                )
            ),
        ],
        fluid=True,
        className="page-shell",
    )


app.layout = html.Div([dcc.Location(id="url"), nav_bar(), html.Div(id="page-content")])


@callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/hypotheses":
        return hypotheses_page()
    return overview_page()


@callback(
    Output("road-risk-map", "figure"),
    Output("road-risk-summary", "children"),
    Output("road-risk-top-table", "children"),
    Input("road-risk-states", "value"),
    Input("road-risk-metric", "value"),
    Input("road-risk-min-count", "value"),
)
def update_road_risk_map(selected_states, metric, min_count):
    filtered = filter_road_danger_data(ROAD_DANGER_DF, selected_states=selected_states, min_count=min_count)
    return (
        build_road_danger_map(
            ROAD_DANGER_DF,
            metric=metric,
            selected_states=selected_states,
            min_count=min_count,
        ),
        build_road_summary_strip(filtered, metric),
        build_road_top_table(filtered, metric),
    )


@callback(
    Output("download-road-html", "data"),
    Input("download-road-html-btn", "n_clicks"),
    State("road-risk-states", "value"),
    State("road-risk-metric", "value"),
    State("road-risk-min-count", "value"),
    prevent_initial_call=True,
)
def download_road_html(n_clicks, selected_states, metric, min_count):
    fig = build_road_danger_map(
        ROAD_DANGER_DF,
        metric=metric,
        selected_states=selected_states,
        min_count=min_count,
    )
    html_str = fig.to_html(full_html=True, include_plotlyjs=True)
    return dict(content=html_str, filename="road_risk_map.html")


@callback(
    Output("download-road-csv", "data"),
    Input("download-road-csv-btn", "n_clicks"),
    State("road-risk-states", "value"),
    State("road-risk-metric", "value"),
    State("road-risk-min-count", "value"),
    prevent_initial_call=True,
)
def download_road_csv(n_clicks, selected_states, metric, min_count):
    filtered = filter_road_danger_data(ROAD_DANGER_DF, selected_states=selected_states, min_count=min_count)
    filtered = filtered.sort_values(metric, ascending=False)
    return dcc.send_data_frame(filtered.to_csv, "road_risk_filtered.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
