# Week 4 Summary - API Deployment & Dashboard

## Deployment Overview
- **API Framework**: Flask REST API
- **Model Served**: LightGBM production model
- **Dashboard**: Interactive web UI
- **Prediction Speed**: ~2–5 ms per request
- **Explainability**: SHAP integrated in API responses

## Key Components

### 1. Flask Prediction API
- `/predict` endpoint for real-time predictions
- Accepts CNC sensor inputs via JSON
- Returns:
  - failure probability
  - prediction label (normal / failure)
  - response time
  - SHAP feature contributions

### 2. Feature Engineering in API
Additional runtime features created during inference:

- Temperature × vibration interaction
- Vibration squared
- Temperature-to-wear ratio
- Vibration–wear interaction
- Temperature deviation from baseline

These engineered signals help capture **machine stress patterns before failure**.

### 3. Interactive Monitoring Dashboard
A web-based dashboard was implemented to simulate industrial monitoring.

**Dashboard Features**

- Adjustable sensor inputs
- Real-time failure prediction
- Risk classification (LOW / MEDIUM / HIGH)
- Failure probability display
- API response time monitoring

This interface demonstrates how maintenance engineers could interact with the predictive system.

### 4. Production Simulation
The deployment simulates a real industrial monitoring pipeline:

Dashboard → API → Model → Prediction → Explanation

The system can generate predictions within **milliseconds**, making it suitable for real-time monitoring environments.

---

Week 4 Progress: 100% Complete!

✅ Day 1: Flask API implementation - DONE  
✅ Day 2: Model loading & prediction endpoint - DONE  
✅ Day 3: Feature engineering in API - DONE  
✅ Day 4: SHAP explainability integration - DONE  
✅ Day 5: Dashboard interface development - DONE  
✅ Day 6: API response monitoring - DONE  
✅ Day 7: End-to-end system testing - DONE  

Current Status: **Project Deployment Complete 🚀**