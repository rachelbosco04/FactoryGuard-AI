# Predictive Maintenance for CNC Machines 🔧

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Advanced machine learning system for predicting CNC machine failures using synthetic sensor data

## 📋 Project Overview

This project implements a complete **predictive maintenance solution** for Computer Numerical Control (CNC) manufacturing equipment. The system analyzes 15 different sensor measurements to predict equipment failures before they occur, enabling proactive maintenance and reducing downtime.

### 🎯 Key Features

- **Synthetic Dataset Generation**: Custom-built realistic CNC machine sensor data
- **Advanced Feature Engineering**: Rolling window statistics, lag features, degradation indices
- **Imbalanced Learning**: SMOTE, class weighting, and precision-recall optimization
- **Model Ensemble**: XGBoost, LightGBM, and Random Forest with hyperparameter tuning
- **Explainable AI**: SHAP analysis for model interpretability
- **Production API**: Flask-based REST API for real-time predictions

## 📊 Dataset Specifications

- **Machines**: 500 CNC machines
- **Duration**: 6 months of hourly sensor readings
- **Records**: ~2.1 million data points
- **Sensors**: 15 different measurements (temperature, vibration, pressure, etc.)
- **Failure Rate**: 2.5% (realistic imbalance)
- **Features**: 18 total (15 sensors + 3 operational settings)

## 🗂️ Project Structure

```
predictive-maintenance-project/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                       # Configuration parameters
│
├── data/
│   ├── raw/                          # Original synthetic data
│   ├── processed/                    # Feature-engineered data
│   ├── external/                     # External reference data
│   └── synthetic_generation/         # Dataset generation scripts
│       └── generate_dataset.py       # Main data generator
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and visualization
│   ├── 02_feature_engineering.ipynb  # Rolling windows, lag features
│   ├── 03_modeling.ipynb             # Model training
│   ├── 04_imbalance_handling.ipynb   # SMOTE and class balancing
│   ├── 05_explainability.ipynb       # SHAP analysis
│   └── 06_deployment_demo.ipynb      # API testing
│
├── src/
│   ├── data/                         # Data processing modules
│   ├── features/                     # Feature engineering
│   ├── models/                       # Model training
│   ├── evaluation/                   # Metrics and explainability
│   ├── imbalance/                    # Imbalance handling
│   └── deployment/                   # API and serving
│
├── models/                           # Saved models
├── reports/                          # Results and figures
├── deployment/                       # Docker and API files
├── tests/                            # Unit tests
└── docs/                             # Documentation
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd predictive-maintenance-project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
cd data/synthetic_generation
python generate_dataset.py
```

This will create:
- `data/raw/train_data.csv` (1.68M records)
- `data/raw/test_data.csv` (0.42M records)
- `data/raw/data_dictionary.txt`

### 3. Run Notebooks

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Follow notebooks 01-06 in sequence.

### 4. Deploy API

```bash
cd deployment
python app.py
```

API will be available at `http://localhost:5000`

## 📈 Project Timeline

### Week 1: Feature Engineering ✅
- Advanced rolling window statistics (6, 12, 24 hour windows)
- Exponential moving averages
- Lag features (t-1, t-2, t-3)
- Rate of change calculations
- Degradation indices

### Week 2: Modeling 🔄
- Baseline models (Logistic Regression, Random Forest)
- Advanced models (XGBoost, LightGBM)
- Hyperparameter tuning with Optuna
- Model ensembling

### Week 3: Explainability 🔄
- SHAP analysis
- Feature importance visualization
- Model interpretation
- Final report generation

### Week 4: Deployment 🔄
- Flask API development
- Latency optimization
- Docker containerization
- API documentation

## 🎯 Performance Metrics

Target metrics for model evaluation:

- **Primary**: Precision-Recall AUC (PR-AUC)
- **Secondary**: Precision, Recall, F1-Score
- **Business**: False Negative Rate (missed failures)

## 🔬 Model Architecture

```
Input (18 features)
    ↓
Feature Engineering (Rolling Windows, Lags)
    ↓
SMOTE (Balance Classes)
    ↓
XGBoost/LightGBM Ensemble
    ↓
Threshold Optimization
    ↓
Output (Failure Probability)
```

## 📚 Key Technologies

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **ML Libraries**: scikit-learn, xgboost, lightgbm
- **Imbalance**: imbalanced-learn (SMOTE)
- **Explainability**: SHAP
- **Deployment**: Flask, Docker
- **Optimization**: Optuna

## 📊 Sample Results

(To be updated after model training)

```
Model Performance:
- PR-AUC: 0.XX
- Precision: 0.XX
- Recall: 0.XX
- F1-Score: 0.XX
```

## 🔌 API Usage

### Predict Failure

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": 42,
    "spindle_temp": 67.5,
    "spindle_vibration": 0.45,
    "motor_current": 15.8,
    ...
  }'
```

### Response

```json
{
  "machine_id": 42,
  "failure_probability": 0.87,
  "predicted_rul": 15,
  "risk_level": "CRITICAL",
  "top_sensors": ["spindle_vibration", "tool_wear", "motor_current"],
  "recommendation": "Schedule immediate maintenance"
}
```

## 🧪 Testing

```bash
pytest tests/
```

## 📖 Documentation

Detailed documentation available in `docs/`:
- Architecture diagram
- API documentation
- Weekly progress reports
- Feature engineering guide

## 🤝 Contributing

This is a course project, but feedback and suggestions are welcome!

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

**Your Name**
- Project for: [Course Name]
- Institution: [Your University]
- Date: February 2026

## 🙏 Acknowledgments

- Dataset inspired by NASA C-MAPSS turbofan engine dataset
- Feature engineering techniques from industry best practices
- SHAP methodology from Lundberg & Lee (2017)

---

**⭐ If you find this project helpful, please star the repository!**
