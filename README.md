# 🚗 US Accident Severity Prediction (2016–2023)

## 📌 Overview
This project predicts the severity of US traffic accidents using machine learning models trained on a dataset of **7.7 million records** collected between 2016 and 2023.

**Course:** DATS 6202 - Machine Learning I  
**University:** George Washington University  
**Team Members:** Nazish Atta, Ilgaz Kusku, Alejandro Gomez  
**Instructor:** Yuxiao (James) Huang  

---

## 📂 Dataset
- **Source:** [US Accidents Dataset – Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size:** 7,728,394 rows × 46 columns
- **Target Variable:** `Severity` (1 = least severe → 4 = most severe)

---

## 🔧 Project Pipeline

### 1. Data Preparation
- Loaded dataset in chunks to handle large memory size
- Removed columns with high missing values (End_Lat, End_Lng, Wind_Chill)
- Fixed outliers in Temperature, Wind Speed, Visibility, Pressure
- Imputed missing values using median and "Unknown"
- Removed rows with invalid timestamps

### 2. Feature Engineering
- Extracted time features: hour, day, month, quarter, season
- Created behavior features: rush hour, weekend, night, business hours
- Calculated accident duration in minutes
- Applied cyclical encoding (sin/cos) for hour, day, and month

### 3. Exploratory Data Analysis (EDA)
- Severity class distribution
- Accidents by year, month, hour, and day of week
- Top states and cities by accident count
- Weather variable distributions
- Correlation heatmap
- Road features vs. severity

---

## 🤖 Models Trained

| Model | Weighted F1 | Train Time | Sample Size |
|---|---|---|---|
| Logistic Regression | 0.5192 | 8.4s | 100,000 |
| Random Forest | 0.7589 | 217.6s | 100,000 |
| **LightGBM** ✅ | **0.7595** | 646.3s | 500,000 |
| MLP Neural Network | 0.7429 | 108.9s | 100,000 |

---

## 🏆 Best Model: LightGBM
- **Weighted F1-Score: 0.7595**
- Best parameters: `learning_rate=0.1`, `num_leaves=63`
- Strongest performance across all severity classes
- Top features: Start location, accident duration, distance, year, weather

---

## 📊 Key Findings
- Severity 2 accounts for **79.67%** of all accidents (heavy class imbalance)
- Most accidents occur during **morning and evening rush hours**
- **California, Florida, and Texas** have the highest accident counts
- **Junctions and traffic signals** are associated with higher severity

---

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, LightGBM
- Matplotlib, Seaborn
- Google Colab + Google Drive

---

## 📁 Repository Structure
