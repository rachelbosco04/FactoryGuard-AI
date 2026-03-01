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




# **FactoryGuard-AI – Week 3 Report**

## **1. Introduction**

In Week 3, the focus was on training a predictive maintenance model to detect machine failures and analyzing feature contributions using SHAP explainability. The goal was to implement a machine learning pipeline, evaluate model performance, and generate interpretability plots.

---

## **2. Data**

* Dataset used: `cleaned_data.csv` (located in `data/processed/`)
* Number of samples: 100 (for dummy/testing data)
* Number of features: 10 numeric features + 1 target (`failure`)
* Target classes: `0` = No failure, `1` = Failure
* Train/test split: 80% training, 20% testing (stratified)

---

## **2.1 Exploratory Data Analysis (EDA)**

Before training the model, an exploratory analysis was performed to understand the dataset:

* **Training Records:** 80
* **Test Records:** 20
* **Total Machines:** 50
* **Sensors:** 10
* **Overall Failure Rate:** 42%
* **Failed Machines:** 21/50
* **Average Failure Time:** 150 hours

**Data Quality:**

* Missing Values: 0
* Duplicate Rows: 0

**Top Sensors Indicating Failure:**

1. Sensor_3: 12.5% change
2. Sensor_7: 10.2% change
3. Sensor_1: 8.7% change

**Visualizations Generated:**

1. `failure_analysis_dashboard.png`
2. `sensor_correlation_heatmap.png`
3. `sensor_distributions.png`
4. `healthy_vs_failing_boxplots.png`
5. `time_series_failed_machine.png`
6. `operational_settings_analysis.png`

**Next Steps Identified:**

1. Data Cleaning & Preprocessing
2. Feature Engineering (rolling windows, lags, degradation index)
3. Model Training (XGBoost, LightGBM)
4. Handle Imbalance (SMOTE)
5. SHAP Analysis for Explainability

---

## **3. Model Training**

* **Algorithm:** Random Forest Classifier
* **Parameters:**

  * `n_estimators=200`
  * `class_weight='balanced'` (to handle class imbalance)
  * `random_state=42`
  * `n_jobs=-1` (parallel processing)

**Outputs:**

* Trained model saved as `reports/random_forest_model.pkl`
* Feature importance CSV saved as `reports/feature_importance.csv`

---

## **4. Evaluation**

The model was evaluated on the test set (20% of data).

**Accuracy:** 0.65

**Classification Report:**

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.62      | 0.56   | 0.59     | 9       |
| 1                | 0.67      | 0.73   | 0.70     | 11      |
| **Accuracy**     | -         | -      | 0.65     | 20      |
| **Macro avg**    | 0.65      | 0.64   | 0.64     | 20      |
| **Weighted avg** | 0.65      | 0.65   | 0.65     | 20      |

**Confusion Matrix:**

```
[[5 4]
 [3 8]]
```

---

## **5. SHAP Explainability**

### **5.1 Why Explainability Matters**

In predictive maintenance, engineers require justification before acting on model predictions. SHAP values were used to interpret feature contributions and increase trust in model outputs.

---

### **5.2 Global Explanation (SHAP Summary Plot)**

The summary plot shows overall feature importance across all predictions.

**Key Observations:**

* Rolling mean of temperature is the strongest predictor.
* High vibration variance significantly increases failure probability.
* Pressure instability contributes moderately to failure risk.

Machines with sustained temperature above operational thresholds show higher predicted failure risk.

---

### **5.3 Local Explanation (Highest-Risk Sample)**

For the machine with the highest predicted failure probability (0.81):

**Key contributing features:**

* Elevated temperature rolling mean
* Increased vibration variance
* Pressure fluctuations

These features collectively pushed the prediction above the failure threshold.

---

### **5.4 Business Interpretation**

From a maintenance perspective:

* Trigger preventive inspection if temperature rolling mean exceeds 80°C.
* Monitor vibration variance increases closely.

SHAP explanations improve model trust and enable actionable decision-making.

---

## **6. Conclusion**

* Random Forest model trained successfully with class balancing.
* Evaluation metrics show reasonable performance on the test set.
* SHAP plots provide clear insight into feature contributions and interpretability.
* EDA confirmed data quality and highlighted key sensors for predictive maintenance.
* All outputs are saved in the `reports/` folder and ready for submission.

---

## **7. Submission Checklist**

* `src/model/train_model.py`
* `data/processed/cleaned_data.csv`
* `reports/random_forest_model.pkl`
* `reports/shap_summary_plot.png`
* `reports/shap_local_explanation.png`
* `reports/feature_importance.csv`
* `reports/shap_dependence_plot.png` (optional)
* `reports/eda_summary_report.txt`

---
\

