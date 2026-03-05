# 🏭 FactoryGuard-AI
### CNC Machine Predictive Maintenance System

FactoryGuard-AI is a predictive maintenance platform that analyzes CNC machine sensor data to detect degradation patterns and predict machine failures **before breakdown occurs**.

The system combines **time-series feature engineering, machine learning, explainable AI (SHAP), and real-time deployment** to simulate a production-ready industrial monitoring system.

---

# 🎯 Objectives

- Predict machine failures in advance  
- Detect degradation patterns in sensor signals  
- Handle extreme class imbalance in failure data  
- Engineer advanced time-series features  
- Provide explainable AI insights using **SHAP**  
- Deploy the model through a **real-time API and dashboard**

---

# 🏗 Dataset

The project uses a **synthetic CNC machine dataset** that simulates realistic industrial telemetry.

### Dataset Characteristics

| Metric | Value |
|------|------|
| Machines | 500 |
| Sensor Records | 2.1M+ |
| Sensors | 15 |
| Duration | 6 months |
| Failure Events | 8 |
| Failure Rate | ~0.4% |

### Sensor Examples

- Spindle temperature  
- Spindle vibration  
- Motor current  
- Tool wear  
- Hydraulic pressure  
- Coolant temperature  
- Feed rate  
- Ambient temperature  

---


---

# 📅 Project Progress

## 📊 Week 1 — Data Exploration & Engineering

**Dataset Overview**

- 500 CNC machines monitored
- 2.1M+ sensor readings
- 6 months of simulated operation
- ~0.4% failure rate

**Key Findings**

- Failures develop gradually over **100–120 hours**
- Degradation patterns appear before breakdown
- Critical sensors identified:
  - spindle_vibration
  - tool_wear
  - motor_current

**Data Cleaning Techniques**

- Forward fill for time-series gaps
- IQR-based outlier capping
- Rolling window noise reduction

**Results**

- Noise reduced by ~18.6%
- Degradation signals preserved
- No data loss

Week 1 Status: **Complete**

---

## 🤖 Week 2 — Model Training

Multiple machine learning models were trained and compared.

**Models Trained**

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Tuned XGBoost
- Tuned LightGBM

**Best Model**

LightGBM (tuned)

| Metric | Value |
|------|------|
| PR-AUC | 0.75+ |
| Prediction Speed | ~2 ms |
| Failure Detection | ~75% |

**Important Features**

- spindle_vibration_lag_1h
- tool_wear_rolling_24h
- degradation_index

Week 2 Status: **Complete**

---

## 🔎 Week 3 — Explainable AI (SHAP)

SHAP was used to interpret model predictions and provide transparency.

**Explainability Outputs**

- SHAP summary plots
- Feature importance rankings
- Local prediction explanations

**Key Insights**

Most influential predictors:

- vibration spikes
- increasing tool wear
- degradation index patterns

SHAP explanations allow engineers to understand **why a failure prediction occurs**, increasing trust in the model.

Week 3 Status: **Complete**

---

## 🚀 Week 4 — Deployment

The trained model was deployed using a **Flask API and interactive dashboard**.


### System Features

- Real-time prediction endpoint
- Sensor input through dashboard
- Failure probability prediction
- Risk classification (LOW / MEDIUM / HIGH)
- Response time monitoring


Week 4 Status: **Complete**

---

# ⚡ System Performance

| Metric | Value |
|------|------|
| Model | LightGBM |
| Prediction Speed | ~2 ms |
| PR-AUC | 0.75+ |
| Failure Detection | ~75% |

---

# 🔮 Future Improvements

- Real-time IoT sensor streaming
- Edge device deployment
- Cloud-based monitoring
- Automated maintenance scheduling
- Integration with industrial monitoring systems


