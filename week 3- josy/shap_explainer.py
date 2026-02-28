import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "reports" / "random_forest_model.pkl"
TEST_PATH  = BASE_DIR / "data" / "processed" / "test_cleaned.csv"
REPORTS_DIR = BASE_DIR / "reports"

# ---------------------------------------------------
# LOAD MODEL AND DATA
# ---------------------------------------------------

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)

drop_cols = ["failure", "hours_to_failure"]

if "timestamp" in test_df.columns:
    drop_cols.append("timestamp")

if "machine_id" in test_df.columns:
    drop_cols.append("machine_id")

X_test = test_df.drop(columns=drop_cols)
y_test = test_df["failure"]

# ---------------------------------------------------
# SHAP EXPLAINER
# ---------------------------------------------------

print("Generating SHAP explanations...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Select class 1 (failure)
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values

# ---------------------------------------------------
# GLOBAL EXPLANATION (Summary Plot)
# ---------------------------------------------------

plt.figure()
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "shap_summary_plot.png")
plt.close()

print("SHAP summary plot saved.")

# ---------------------------------------------------
# LOCAL EXPLANATION (Most Risky Case)
# ---------------------------------------------------

# Get prediction probabilities
probs = model.predict_proba(X_test)[:, 1]
max_prob_index = probs.argmax()

print(f"Highest predicted failure probability: {probs[max_prob_index]:.2f}")

# Create explanation object for single row
single_explanation = shap.Explanation(
    values=shap_values_class1[max_prob_index],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[max_prob_index],
    feature_names=X_test.columns
)

plt.figure()
shap.plots.waterfall(single_explanation, show=False)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "shap_local_explanation.png")
plt.close()

print("Local SHAP explanation saved.")
print("All SHAP outputs saved inside reports folder ✅")