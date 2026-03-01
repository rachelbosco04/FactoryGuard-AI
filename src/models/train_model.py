import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up from src/model to FactoryGuard-AI
DATA_PATH = PROJECT_ROOT / "data/processed/cleaned_data.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

# -------------------------------
# Main Training Function
# -------------------------------
def main():
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Loading data from: {DATA_PATH}")

    # -------------------------------
    # Load Data
    # -------------------------------
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError(f"CSV is empty: {DATA_PATH}")

    print("\nPreparing features...")
    X = df.drop("failure", axis=1)
    y = df["failure"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nClass Distribution in Training Set:")
    print(y_train.value_counts())

    # -------------------------------
    # Train Random Forest
    # -------------------------------
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Save model
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = REPORTS_DIR / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved at: {model_path}")

    # -------------------------------
    # Evaluation
    # -------------------------------
    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # -------------------------------
    # SHAP Explainability
    # -------------------------------
    print("\nGenerating SHAP explanations...")
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    explainer = shap.Explainer(model, X_sample)
    shap_values_sample = explainer(X_sample)

    # ---------------------------
    # Global Summary Plot
    # ---------------------------
    plt.figure()
    shap.summary_plot(
        shap_values_sample.values[:, :, 1],  # Class 1 (failure)
        X_sample,
        show=False
    )
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_plot.png")
    plt.close()
    print("SHAP summary plot saved.")

    # ---------------------------
    # Feature Importance CSV
    # ---------------------------
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print("Feature importance CSV saved.")

    # ---------------------------
    # Local Explanation (Highest Risk Sample)
    # ---------------------------
    probs = model.predict_proba(X_test)[:, 1]
    highest_risk_index = probs.argmax()
    print("\nHighest predicted failure probability:", probs[highest_risk_index])

    single_sample = X_test.iloc[[highest_risk_index]]
    single_shap = explainer(single_sample)

    plt.figure()
    shap.plots.waterfall(single_shap[0, :, 1], show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_local_explanation.png")
    plt.close()
    print("Local SHAP explanation saved.")

    # ---------------------------
    # Optional: Dependence Plot (safe for small datasets)
    # ---------------------------
    try:
        if X_sample.shape[0] > 1:
            feature = X_sample.columns[0]  # first feature for dependence plot
            shap.dependence_plot(
                feature,
                shap_values_sample.values[:, :, 1] if hasattr(shap_values_sample, "values") else shap_values_sample,
                X_sample,
                show=False
            )
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "shap_dependence_plot.png")
            plt.close()
            print("SHAP dependence plot saved.")
        else:
            print("Skipped SHAP dependence plot (dataset too small).")
    except Exception as e:
        print(f"Skipped SHAP dependence plot due to error: {e}")

    print("\nAll outputs saved inside reports folder ✅")


# -------------------------------
# Run Script
# -------------------------------
if __name__ == "__main__":
    main()