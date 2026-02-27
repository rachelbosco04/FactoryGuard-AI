""" 
Hyperparameter Tuning - week 2
========================================

Optimizes XGBoost model using optuna
Metric - PR-AUC (Average precision score)
Handles severe class imbalance

"""

#importing libraries

import os
import joblib
import optuna
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

#load the data

train = pd.read_csv("data/processed/train_cleaned.csv")
test = pd.read_csv("data/processed/test_cleaned.csv")

X_train = train.drop("failure",axis=1)
y_train= train["failure"]

X_test= test.rop("failure",axis=1)
y_test = test["failure"]



#function objective

def objective(trial,X,y):
    
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)
    
    model=XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth = trial.suggest_int("max_depth", 3, 8),
        learning_rate= trial.suggest_float("subsample", 0.01, 0.2),
        subsample=trial.suggest_float("subsmple", 0.7,1.0),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )
    
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    
    score=cross_val_score(
        
        model,
        X_train,
        y_train,
        scoring="average-precision",
        cv=cv
    ).mean()
    
    return score

#hyperparameter tuning


print("\nStarting Hyperparameter Tuning..")

study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=20)

print("/nBest Parameters:")

print(study.best_params)

print("/nBest Cross-validation PR-AUC:")
print(study.best_value)



# 4️⃣ TRAIN FINAL MODEL
# ==========================================

print("\nTraining final model...")

best_model = XGBClassifier(
    **study.best_params,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    eval_metric="logloss",
    random_state=42
)

best_model.fit(X_train, y_train)


# ==========================================
# 5️⃣ FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating on Test Set...")

y_probs = best_model.predict_proba(X_test)[:, 1]
test_pr_auc = average_precision_score(y_test, y_probs)

print("Test PR-AUC:", test_pr_auc)


# ==========================================
# 6️⃣ SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

print("\nModel saved successfully in models/best_model.pkl ✅")
print("\nWeek 2 Completed 🚀")