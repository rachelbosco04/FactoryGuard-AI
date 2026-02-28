# 📘 Week 3 – Model Explainability (SHAP Integration)

##  Objective

The goal of Week 3 was to improve **model trust and adoption** by integrating explainability into the predictive maintenance system.

Since the maintenance engineer needs to understand *why* a machine failure was predicted, SHAP (SHapley Additive exPlanations) was implemented to generate both:

* Global explanations (feature importance across dataset)
* Local explanations (why a specific machine was predicted to fail)

---

## Implementation Overview

### 1️⃣ Model Used

* Random Forest Classifier
* Trained on processed sensor data
* Class imbalance handled using `class_weight="balanced"`

---

### 2️⃣ SHAP Integration

SHAP was added inside:

```
src/models/train_model.py
```

We used:

```python
explainer = shap.Explainer(model, X_sample)
```

to compute Shapley values.

---

## Outputs Generated

All outputs are stored inside the `reports/` folder.

### ✅ 1. SHAP Summary Plot

**File:** `shap_summary_plot.png`

Purpose:

* Shows which features most influence failure prediction
* Displays global feature importance
* Helps identify critical sensor metrics (e.g., temperature, vibration trends)

---

### ✅ 2. SHAP Local Explanation

**File:** `shap_local_explanation.png`

Purpose:

* Explains why the model predicted failure for the highest-risk machine
* Shows how each feature contributed positively or negatively
* Enables engineers to understand the exact reasoning behind prediction

Example interpretation:

> "The rolling mean of temperature increased significantly and vibration variance spiked, increasing failure probability."

---

## 📈 Model Evaluation Summary

* Accuracy: ~99.99%
* Severe class imbalance observed (very few failure cases)
* Confusion matrix shows difficulty predicting rare failure class

⚠️ Note: Accuracy is high due to imbalance. Future improvements may include:

* SMOTE
* Threshold tuning
* More failure data collection

---

## Business Impact

By integrating SHAP:

* The model is no longer a black-box
* Engineers can understand failure drivers
* Increases confidence in AI-based maintenance decisions
* Supports real-world deployment readiness

---

## Conclusion

Week 3 successfully enhanced the predictive maintenance system by:

* Adding explainability
* Generating interpretable visual reports
* Improving model transparency
* Making AI decisions actionable


