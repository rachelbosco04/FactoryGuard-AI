import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/final_model.pkl")

# Load processed data
X = pd.read_csv("data/processed/train_cleaned.csv")
X = X.drop("target", axis=1)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# -------------------------
# GLOBAL EXPLANATION
# -------------------------
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("reports/shap_summary.png")

# -------------------------
# LOCAL EXPLANATION (Single Prediction)
# -------------------------
plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X.iloc[0],
        feature_names=X.columns
    ),
    show=False
)

plt.savefig("reports/shap_local_explanation.png")