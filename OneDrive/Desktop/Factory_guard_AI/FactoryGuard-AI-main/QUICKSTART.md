# Quick Start Guide 🚀

## Complete Setup in 5 Steps

### Step 1: Generate Your Dataset

```bash
cd data/synthetic_generation
python generate_dataset.py
```

**Expected output:**
```
Generating synthetic dataset...
Machines: 500
Hours per machine: 4320
Target failure rate: 2.5%
--------------------------------------------------
Generated data for 50/500 machines...
Generated data for 100/500 machines...
...
--------------------------------------------------
✅ Dataset generation complete!
Total records: 2,160,000
Failed machines: 13/500 (2.6%)
Failure records: 54,000 (2.5%)

✅ Dataset generation complete! Files saved:
   📁 ../raw/train_data.csv
   📁 ../raw/test_data.csv
   📁 ../raw/full_dataset.csv
   📄 ../raw/data_dictionary.txt
```

### Step 2: Explore the Data

Open the first notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Key visualizations to create:
- Failure distribution across machines
- Sensor correlations heatmap
- Time-series plots of key sensors
- Healthy vs failing machine comparisons

### Step 3: Engineer Features (Week 1)

```bash
cd src/features
python feature_engineering.py
```

This creates:
- Rolling window statistics (6, 12, 24, 48 hours)
- Lag features (t-1, t-2, t-3, t-6, t-12)
- Exponential moving averages
- Rate of change features
- Degradation indices
- Interaction features

**Pro tip:** Review `config.yaml` to adjust feature engineering parameters!

### Step 4: Train Models (Week 2)

Open the modeling notebook:

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

Models to train:
1. Baseline: Logistic Regression, Random Forest
2. Advanced: XGBoost, LightGBM
3. With SMOTE for imbalance handling
4. Hyperparameter tuning with Optuna

### Step 5: Deploy API (Week 4)

```bash
cd deployment
python app.py
```

Test your API:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": 42,
    "spindle_temp": 67.5,
    "spindle_vibration": 0.45,
    "motor_current": 15.8
  }'
```

---

## Project Timeline

### Week 1: Feature Engineering ✅
**Deliverable:** Processed datasets with engineered features

Tasks:
- [ ] Generate synthetic dataset
- [ ] EDA and visualization
- [ ] Rolling window features
- [ ] Lag features
- [ ] Degradation indices
- [ ] Save processed data

**Files to submit:**
- `data/processed/train_features.csv`
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_feature_engineering.ipynb`

### Week 2: Modeling 🔄
**Deliverable:** Trained models with hyperparameter tuning

Tasks:
- [ ] Baseline models (Logistic Regression, RF)
- [ ] XGBoost/LightGBM training
- [ ] SMOTE for imbalance
- [ ] Hyperparameter tuning (Optuna)
- [ ] Model comparison
- [ ] Save best model

**Files to submit:**
- `models/best_model.pkl`
- `notebooks/03_modeling.ipynb`
- `notebooks/04_imbalance_handling.ipynb`
- `reports/metrics/model_comparison.csv`

### Week 3: Explainability 🔄
**Deliverable:** SHAP analysis and final report

Tasks:
- [ ] SHAP value calculation
- [ ] Feature importance plots
- [ ] Individual prediction explanations
- [ ] Create visualizations
- [ ] Write final report

**Files to submit:**
- `notebooks/05_explainability.ipynb`
- `reports/figures/shap_plots/`
- `reports/final_report.pdf`

### Week 4: Deployment 🔄
**Deliverable:** Production-ready Flask API

Tasks:
- [ ] Build Flask API
- [ ] Input validation
- [ ] Error handling
- [ ] API documentation
- [ ] Latency optimization
- [ ] Docker containerization

**Files to submit:**
- `deployment/app.py`
- `deployment/Dockerfile`
- `docs/api_documentation.md`

---

## Tips for Standing Out

### 1. **Dataset Quality**
- Your custom dataset is already unique ✅
- Add edge cases (sudden failures, sensor malfunctions)
- Document your data generation logic thoroughly

### 2. **Feature Engineering**
```python
# Go beyond basic features:
✅ Rolling statistics with multiple windows
✅ Exponential moving averages
✅ Interaction features (temperature × vibration)
✅ Degradation index (composite health score)
✅ Time-based features (hours since last maintenance)
```

### 3. **Model Ensemble**
```python
# Don't use just one model:
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('rf', rf_model)
], voting='soft')
```

### 4. **Advanced SHAP**
```python
# Create interactive visualizations:
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Waterfall plots for individual predictions
shap.plots.waterfall(shap_values[0])

# Force plots for interactive exploration
shap.plots.force(explainer.expected_value, shap_values[0])
```

### 5. **Production API**
```python
# Add these features:
✅ Input validation (Pydantic)
✅ Error handling
✅ Logging
✅ Health check endpoint
✅ API documentation (Swagger)
✅ Model versioning
✅ Latency monitoring
```

---

## Troubleshooting

### Issue: Dataset generation is slow
**Solution:** Reduce `n_machines` in config.yaml for testing

### Issue: Out of memory during feature engineering
**Solution:** Process machines in batches

### Issue: Models take too long to train
**Solution:** Reduce `n_trials` in Optuna config

---

## Need Help?

- Check `docs/` for detailed documentation
- Review example notebooks in `notebooks/`
- Consult `config.yaml` for all parameters

**Good luck! 🎯**
