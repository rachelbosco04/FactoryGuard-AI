# 🏭 FactoryGuard-AI  
### CNC Machine Predictive Maintenance Engine

---

## 📌 Project Overview

FactoryGuard-AI is an industrial predictive maintenance system designed to anticipate CNC machine failures using time-series sensor analytics and machine learning.

The system analyzes multi-sensor telemetry (temperature, vibration, pressure, wear, etc.) to detect degradation patterns and predict failures **before catastrophic breakdown**, enabling scheduled maintenance and minimizing downtime.

---

## 🎯 Objectives

- Predict machine failure in advance  
- Preserve degradation signals while reducing noise  
- Handle extreme class imbalance  
- Engineer advanced time-series features  
- Provide explainable AI insights (SHAP)  
- Simulate real-world industrial deployment  

---

## 🏗️ Dataset

This project uses a **synthetic CNC Machine dataset** generated with realistic industrial behavior:

- 500 machines simulated  
- 6 months of operation per machine  
- 15 sensor channels  
- Controlled failure patterns  
- Gradual degradation prior to failure  
- Rare failure events (~1.6% machines)

---

## 📊 Data Characteristics

| Metric | Value |
|--------|-------|
| Total Records | 2,148,873 |
| Machines | 500 |
| Features | 23 |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Failure Events | 8 |

---

## 🔍 Week 1 Progress

### ✅ Dataset Generation  
✔ Realistic sensor simulation  
✔ Controlled degradation patterns  
✔ Rare failure injection  

---

### ✅ Exploratory Data Analysis (EDA)

- Sensor distribution analysis  
- Failure pattern visualization  
- Correlation heatmaps  
- Healthy vs failing comparisons  
- Time-series degradation tracking  

**Key Insight:**  
Degradation signals visible ~120 hours before failure.

---

### ✅ Data Cleaning & Preprocessing

Pipeline:

Raw Data →  
Duplicate Removal →  
Missing Value Handling →  
Outlier Capping (IQR) →  
Noise Reduction (Rolling Window) →  
Cleaned Dataset

**Results:**

- Noise reduced by ~18.6%  
- 100% degradation signals preserved  
- 0% data loss  

---

## 🔧 Cleaning Techniques Used

- Forward Fill (time-series safe)  
- IQR-based outlier capping  
- Rolling window smoothing (5-hour window)  

---

## 📁 Project Structure

FactoryGuard-AI/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ └── 02_data_cleaning.ipynb
│
├── src/
│ ├── feature_engineering.py
│ ├── preprocessing.py
│ └── modeling.py
│
├── reports/
│ └── figures/
│
├── models/
│
├── requirements.txt
└── README.md
