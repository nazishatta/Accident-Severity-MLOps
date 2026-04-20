# Tableau EDA Plan

This folder is set up to make the EDA look better in Tableau without dragging the raw CSV straight into a workbook.

## Files

Run:

```bash
python3 build_tableau_exports.py
```

This creates `/Applications/GitHub/US Accidents (2016-2023)/tableau_exports` with:

- `01_master_sample.csv`  
  Main cleaned sample for flexible Tableau work
- `02_accidents_by_hour.csv`  
  For a clean hourly line chart
- `03_hour_by_severity.csv`  
  For hour vs severity comparison
- `04_accidents_by_month.csv`  
  For monthly trend
- `05_weather_by_severity.csv`  
  For grouped/stacked weather comparison
- `06_state_summary.csv`  
  For filled map or ranked bars by state
- `07_numeric_correlation.csv`  
  For a correlation heatmap
- `08_daynight_by_severity.csv`  
  For day/night severity split
- `09_source_by_severity.csv`  
  For checking whether source matters
- `10_map_sample.csv`  
  For a map layer without overloading Tableau

## Recommended Tableau Story

### Dashboard 1: Dataset Shape
- Severity distribution
- Accident count by hour
- Accident count by month
- Map density from `10_map_sample.csv`

### Dashboard 2: Weather Context
- Weather group vs severity
- Day/Night vs severity
- Visibility or precipitation distribution from `01_master_sample.csv`
- Correlation heatmap from `07_numeric_correlation.csv`

### Dashboard 3: Geography and Interpretation
- State summary map from `06_state_summary.csv`
- Top states by accident count
- Severe rate by state

### Dashboard 4: Hypothesis and Modeling Support
- Source vs severity from `09_source_by_severity.csv`
- Weather group vs severity
- Hour vs severity
- A text box explaining leakage exclusions:
  - do not use post-event fields
  - do not use identifier-like fields as main drivers
  - use pre-accident context for modeling

## Best Tableau Chart Choices

- `02_accidents_by_hour.csv` -> line chart
- `03_hour_by_severity.csv` -> grouped line or grouped bars
- `05_weather_by_severity.csv` -> stacked or grouped bar chart
- `06_state_summary.csv` -> filled map + ranked bar chart
- `07_numeric_correlation.csv` -> highlight table / heatmap
- `10_map_sample.csv` -> symbol map or density map

## Visual Style Suggestions

- Use one calm palette:
  - deep teal
  - muted blue
  - warm coral as emphasis
- Avoid rainbow color scales
- Keep one dashboard title and one short subtitle
- Put insights as text callouts next to the strongest charts
- Do not place too many worksheets in one dashboard

## Why this is better than raw Tableau EDA

The raw dataset is too noisy and too wide. These exports keep the EDA focused on:

- severity balance
- time pattern
- weather pattern
- geography
- hypothesis support

That gives you a more presentation-ready Tableau workbook much faster.
