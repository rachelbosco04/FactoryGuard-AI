import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_recall_curve, auc, make_scorer
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """Tune hyperparameters using Optuna"""
    
    def __init__(self):
        self.best_params = {}
        self.best_models = {}
        self.study_results = {}
        
    def load_data(self, train_path):
        """Load training data"""
        print("="*70)
        print("LOADING DATA FOR HYPERPARAMETER TUNING")
        print("="*70)
        
        print(f"\nLoading training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"✅ Training data loaded: {train_df.shape}")
        
        return train_df
    
    def prepare_features(self, df):
        """Prepare features and target"""
        
        # Columns to exclude
        exclude_cols = [
            'machine_id', 'timestamp', 'hour_of_operation',
            'failure', 'hours_to_failure'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df['failure'].copy()
        
        # Handle NaN and inf
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        return X, y
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data by machine_id"""
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
        
        unique_machines = df['machine_id'].unique()
        train_machines, val_machines = train_test_split(
            unique_machines, test_size=test_size, random_state=random_state
        )
        
        train_df = df[df['machine_id'].isin(train_machines)]
        val_df = df[df['machine_id'].isin(val_machines)]
        
        print(f"\n✅ Training machines: {len(train_machines)}")
        print(f"✅ Validation machines: {len(val_machines)}")
        print(f"✅ Training records: {len(train_df):,}")
        print(f"✅ Validation records: {len(val_df):,}")
        
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        
        return X_train, X_val, y_train, y_val
    
    def pr_auc_score(self, y_true, y_pred_proba):
        """Calculate PR-AUC score"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return auc(recall, precision)
    
    def tune_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Tune XGBoost hyperparameters with Optuna"""
        print("\n" + "="*70)
        print("TUNING XGBOOST HYPERPARAMETERS")
        print("="*70)
        print(f"\n🔍 Running {n_trials} optimization trials...")
        print("⏱️  This may take 15-30 minutes...")
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        def objective(trial):
            """Objective function for Optuna"""
            
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
                'eval_metric': 'aucpr'
            }
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            pr_auc = self.pr_auc_score(y_val, y_pred_proba)
            
            return pr_auc
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name='xgboost_tuning'
        )
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(f"Trial {trial.number}: PR-AUC = {trial.value:.4f}")
            ]
        )
        
        # Store results
        self.best_params['xgboost'] = study.best_params
        self.study_results['xgboost'] = study
        
        # Print results
        print("\n" + "="*70)
        print("XGBOOST TUNING RESULTS")
        print("="*70)
        print(f"\n🏆 Best PR-AUC: {study.best_value:.4f}")
        print(f"\n📊 Best Parameters:")
        for param, value in study.best_params.items():
            print(f"   {param:20s}: {value}")
        
        # Train final model with best params
        print("\n🔄 Training final model with best parameters...")
        best_params = study.best_params.copy()
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'eval_metric': 'aucpr'
        })
        
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10
        )
        
        self.best_models['xgboost'] = best_model
        print("✅ Final model trained!")
        
        return best_model, study.best_params
    
    def tune_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Tune LightGBM hyperparameters with Optuna"""
        print("\n" + "="*70)
        print("TUNING LIGHTGBM HYPERPARAMETERS")
        print("="*70)
        print(f"\n🔍 Running {n_trials} optimization trials...")
        print("⏱️  This may take 15-30 minutes...")
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        def objective(trial):
            """Objective function for Optuna"""
            
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='average_precision',
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            pr_auc = self.pr_auc_score(y_val, y_pred_proba)
            
            return pr_auc
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name='lightgbm_tuning'
        )
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(f"Trial {trial.number}: PR-AUC = {trial.value:.4f}")
            ]
        )
        
        # Store results
        self.best_params['lightgbm'] = study.best_params
        self.study_results['lightgbm'] = study
        
        # Print results
        print("\n" + "="*70)
        print("LIGHTGBM TUNING RESULTS")
        print("="*70)
        print(f"\n🏆 Best PR-AUC: {study.best_value:.4f}")
        print(f"\n📊 Best Parameters:")
        for param, value in study.best_params.items():
            print(f"   {param:20s}: {value}")
        
        # Train final model with best params
        print("\n🔄 Training final model with best parameters...")
        best_params = study.best_params.copy()
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        })
        
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='average_precision',
            callbacks=[lgb.log_evaluation(period=10)]
        )
        
        self.best_models['lightgbm'] = best_model
        print("✅ Final model trained!")
        
        return best_model, study.best_params
    
    def save_results(self, output_dir='models'):
        """Save tuned models and parameters"""
        print("\n" + "="*70)
        print("SAVING TUNED MODELS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.best_models.items():
            filename = f"{model_name}_tuned.pkl"
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"✅ Saved model: {filepath}")
        
        # Save best parameters
        params_path = os.path.join(output_dir, 'best_hyperparameters.pkl')
        joblib.dump(self.best_params, params_path)
        print(f"✅ Saved parameters: {params_path}")
        
        # Save as readable JSON
        import json
        params_json_path = os.path.join(output_dir, 'best_hyperparameters.json')
        with open(params_json_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        print(f"✅ Saved parameters (JSON): {params_json_path}")


def main():
    """Main execution"""
    
    print("="*70)
    print("HYPERPARAMETER TUNING WITH OPTUNA - WEEK 2")
    print("="*70)
    
    # Initialize tuner
    tuner = HyperparameterTuner()
    
    # Load data
    train_path = 'data/processed/train_features.csv'
    
    if not os.path.exists(train_path):
        print(f"❌ ERROR: Cannot find {train_path}")
        return
    
    print(f"✅ Found data file: {train_path}")
    train_df = tuner.load_data(train_path)
    
    # Split data
    X_train, X_val, y_train, y_val = tuner.split_data(train_df, test_size=0.2)
    
    # Tune XGBoost (50 trials ~ 15-20 minutes)
    print("\n" + "🎯 STARTING XGBOOST TUNING")
    xgb_model, xgb_params = tuner.tune_xgboost(
        X_train, y_train, X_val, y_val, 
        n_trials=50
    )
    
    # Tune LightGBM (50 trials ~ 15-20 minutes)
    print("\n" + "🎯 STARTING LIGHTGBM TUNING")
    lgb_model, lgb_params = tuner.tune_lightgbm(
        X_train, y_train, X_val, y_val, 
        n_trials=50
    )
    
    # Save results
    tuner.save_results(output_dir='models')
    
    # Final summary
    print("\n" + "="*70)
    print("✅ HYPERPARAMETER TUNING COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"   XGBoost Best PR-AUC:  {tuner.study_results['xgboost'].best_value:.4f}")
    print(f"   LightGBM Best PR-AUC: {tuner.study_results['lightgbm'].best_value:.4f}")
    print("\n📁 Next Steps:")
    print("   1. Compare all models (baseline + advanced + tuned)")
    print("   2. Select best model for production")
    print("   3. Create final evaluation report")
    print("="*70)


if __name__ == "__main__":
    main()