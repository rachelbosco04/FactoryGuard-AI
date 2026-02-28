import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class AdvancedModelTrainer:
    """Train and evaluate XGBoost and LightGBM models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_data(self, train_path):
        """Load training data"""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        print(f"\nLoading training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"✅ Training data loaded: {train_df.shape}")
        
        return train_df
    
    def prepare_features(self, df):
        """Prepare features and target"""
        
        # Columns to exclude from features
        exclude_cols = [
            'machine_id', 'timestamp', 'hour_of_operation',
            'failure', 'hours_to_failure'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df['failure'].copy() if 'failure' in df.columns else None
        
        # Handle NaN values
        print(f"\n🔍 Checking for NaN values...")
        nan_count_before = X.isnull().sum().sum()
        print(f"   NaN values found: {nan_count_before:,}")
        
        if nan_count_before > 0:
            print(f"   Filling NaN with 0...")
            X = X.fillna(0)
            nan_count_after = X.isnull().sum().sum()
            print(f"   ✅ NaN values after filling: {nan_count_after}")
        
        # Check for infinite values
        inf_count = np.isinf(X).sum().sum()
        if inf_count > 0:
            print(f"   Replacing {inf_count} infinite values with 0...")
            X = X.replace([np.inf, -np.inf], 0)
        
        print(f"\n✅ Features prepared: {X.shape}")
        print(f"✅ Feature columns: {len(feature_cols)}")
        
        if y is not None:
            print(f"✅ Target distribution:")
            print(f"   Normal: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
            print(f"   Failure: {(y==1).sum()} ({(y==1).mean()*100:.2f}%)")
        
        return X, y
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data by machine_id to prevent data leakage"""
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
        
        # Get unique machines
        unique_machines = df['machine_id'].unique()
        
        # Split machines (not records)
        train_machines, val_machines = train_test_split(
            unique_machines, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Create train and validation sets
        train_df = df[df['machine_id'].isin(train_machines)]
        val_df = df[df['machine_id'].isin(val_machines)]
        
        print(f"\n✅ Split by machine_id (prevents data leakage)")
        print(f"   Training machines: {len(train_machines)}")
        print(f"   Validation machines: {len(val_machines)}")
        print(f"   Training records: {len(train_df):,}")
        print(f"   Validation records: {len(val_df):,}")
        
        # Prepare features
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        
        return X_train, X_val, y_train, y_val
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*70)
        print("TRAINING XGBOOST")
        print("="*70)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"\n⚖️  Class imbalance ratio: {scale_pos_weight:.2f}")
        print(f"   Using scale_pos_weight to handle imbalance")
        
        # Initialize model with good defaults
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='aucpr'
        )
        
        # Train with early stopping
        print("\n🔄 Training XGBoost...")
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10
        )
        print("✅ Training complete!")
        
        # Store model
        self.models['xgboost'] = xgb_model
        
        return xgb_model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("\n" + "="*70)
        print("TRAINING LIGHTGBM")
        print("="*70)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"\n⚖️  Class imbalance ratio: {scale_pos_weight:.2f}")
        
        # Initialize model with good defaults
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Train with early stopping
        print("\n🔄 Training LightGBM...")
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='average_precision',
            callbacks=[lgb.log_evaluation(period=10)]
        )
        print("✅ Training complete!")
        
        # Store model
        self.models['lightgbm'] = lgb_model
        
        return lgb_model
    
    def evaluate_model(self, model, X_val, y_val, model_name):
        """Evaluate model performance"""
        print(f"\n{'='*70}")
        print(f"EVALUATING {model_name.upper()}")
        print("="*70)
        
        # Get predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        # Calculate PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Calculate ROC-AUC
        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba)
        except:
            roc_auc = 0.0
        
        # Store results
        results = {
            'model_name': model_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"\n📊 Performance Metrics:")
        print(f"   PR-AUC:    {pr_auc:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"   TN: {cm[0,0]:6,} | FP: {cm[0,1]:6,}")
        print(f"   FN: {cm[1,0]:6,} | TP: {cm[1,1]:6,}")
        
        # Classification Report
        print(f"\n📊 Classification Report:")
        print(classification_report(y_val, y_pred, zero_division=0))
        
        return results
    
    def get_feature_importance(self, model, model_name, X_train, top_n=20):
        """Get and display feature importance"""
        print(f"\n{'='*70}")
        print(f"FEATURE IMPORTANCE - {model_name.upper()}")
        print("="*70)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = X_train.columns
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Display top features
            print(f"\n🔝 Top {top_n} Most Important Features:")
            print(importance_df.head(top_n).to_string(index=False))
            
            return importance_df
        else:
            print("⚠️  Model does not have feature_importances_ attribute")
            return None
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{model_name}_advanced.pkl"
            filepath = os.path.join(output_dir, filename)
            
            joblib.dump(model, filepath)
            print(f"✅ Saved: {filepath}")
        
        # Save results
        results_path = os.path.join(output_dir, 'advanced_results.pkl')
        joblib.dump(self.results, results_path)
        print(f"✅ Saved: {results_path}")
    
    def compare_models(self):
        """Compare advanced models"""
        print("\n" + "="*70)
        print("ADVANCED MODEL COMPARISON")
        print("="*70)
        
        comparison_df = pd.DataFrame([
            {
                'Model': results['model_name'],
                'PR-AUC': results['pr_auc'],
                'ROC-AUC': results['roc_auc'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            }
            for results in self.results.values()
        ])
        
        # Sort by PR-AUC (most important for imbalanced data)
        comparison_df = comparison_df.sort_values('PR-AUC', ascending=False)
        
        print("\n📊 Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Best model
        best_model = comparison_df.iloc[0]['Model']
        best_pr_auc = comparison_df.iloc[0]['PR-AUC']
        print(f"\n🏆 Best Advanced Model: {best_model} (PR-AUC: {best_pr_auc:.4f})")
        
        return comparison_df


def main():
    """Main execution"""
    
    print("="*70)
    print("ADVANCED MODEL TRAINING - WEEK 2")
    print("="*70)
    
    # Initialize trainer
    trainer = AdvancedModelTrainer()
    
    # Load data
    train_path = 'data/processed/train_features.csv'
    
    if not os.path.exists(train_path):
        print(f"❌ ERROR: Cannot find {train_path}")
        return
    
    print(f"✅ Found data file: {train_path}")
    train_df = trainer.load_data(train_path)
    
    # Split data
    X_train, X_val, y_train, y_val = trainer.split_data(train_df, test_size=0.2)
    
    # Train XGBoost
    xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    xgb_results = trainer.evaluate_model(xgb_model, X_val, y_val, 'xgboost')
    xgb_importance = trainer.get_feature_importance(xgb_model, 'xgboost', X_train)
    
    # Train LightGBM
    lgb_model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    lgb_results = trainer.evaluate_model(lgb_model, X_val, y_val, 'lightgbm')
    lgb_importance = trainer.get_feature_importance(lgb_model, 'lightgbm', X_train)
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Save models
    trainer.save_models(output_dir='models')
    
    # Save feature importance
    if xgb_importance is not None:
        xgb_importance.to_csv('models/xgboost_feature_importance.csv', index=False)
        print("✅ Saved: models/xgboost_feature_importance.csv")
    
    if lgb_importance is not None:
        lgb_importance.to_csv('models/lightgbm_feature_importance.csv', index=False)
        print("✅ Saved: models/lightgbm_feature_importance.csv")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ADVANCED MODEL TRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Next Steps:")
    print("   1. Tune hyperparameters with Optuna")
    print("   2. Compare with baseline models")
    print("   3. Select best model for production")
    print("="*70)


if __name__ == "__main__":
    main()