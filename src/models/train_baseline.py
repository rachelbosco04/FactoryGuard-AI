import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class BaselineModelTrainer:
    """Train and evaluate baseline models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_data(self, train_path, test_path=None):
        """Load training and test data"""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load training features
        print(f"\nLoading training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"✅ Training data loaded: {train_df.shape}")
        
        # Load test features if provided
        if test_path:
            print(f"\nLoading test data from: {test_path}")
            test_df = pd.read_csv(test_path)
            print(f"✅ Test data loaded: {test_df.shape}")
            
            # Prepare test data
            X_test, y_test = self.prepare_features(test_df)
            
            return train_df, test_df, X_test, y_test
        
        return train_df, None, None, None
    
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
        """
        Split data ensuring same machine doesn't appear in both train/val
        """
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
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression baseline"""
        print("\n" + "="*70)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*70)
        
        # Initialize model
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',  # Handle imbalance
            n_jobs=-1
        )
        
        # Train
        print("\n🔄 Training Logistic Regression...")
        lr_model.fit(X_train, y_train)
        print("✅ Training complete!")
        
        # Store model
        self.models['logistic_regression'] = lr_model
        
        return lr_model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest baseline"""
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        
        # Initialize model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Train
        print("\n🔄 Training Random Forest...")
        rf_model.fit(X_train, y_train)
        print("✅ Training complete!")
        
        # Store model
        self.models['random_forest'] = rf_model
        
        return rf_model
    
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
        
        # Store results
        results = {
            'model_name': model_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pr_auc': pr_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"\n📊 Performance Metrics:")
        print(f"   PR-AUC:    {pr_auc:.4f}")
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
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{model_name}_baseline.pkl"
            filepath = os.path.join(output_dir, filename)
            
            joblib.dump(model, filepath)
            print(f"✅ Saved: {filepath}")
        
        # Save results
        results_path = os.path.join(output_dir, 'baseline_results.pkl')
        joblib.dump(self.results, results_path)
        print(f"✅ Saved: {results_path}")
    
    def compare_models(self):
        """Compare baseline models"""
        print("\n" + "="*70)
        print("BASELINE MODEL COMPARISON")
        print("="*70)
        
        comparison_df = pd.DataFrame([
            {
                'Model': results['model_name'],
                'PR-AUC': results['pr_auc'],
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
        print(f"\n🏆 Best Baseline Model: {best_model}")
        
        return comparison_df


def main():
    """Main execution"""
    
    print("="*70)
    print("BASELINE MODEL TRAINING - WEEK 2")
    print("="*70)
    
    # Initialize trainer
    trainer = BaselineModelTrainer()
    
    # Use correct path from project root
    train_path = 'data/processed/train_features.csv'
    
    # Verify file exists
    if not os.path.exists(train_path):
        print(f"❌ ERROR: Cannot find {train_path}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    print(f"✅ Found data file: {train_path}")
    
    # Load data
    train_df, _, _, _ = trainer.load_data(train_path=train_path)
    
    # Split data
    X_train, X_val, y_train, y_val = trainer.split_data(train_df, test_size=0.2)
    
    # Train Logistic Regression
    lr_model = trainer.train_logistic_regression(X_train, y_train)
    lr_results = trainer.evaluate_model(lr_model, X_val, y_val, 'logistic_regression')
    
    # Train Random Forest
    rf_model = trainer.train_random_forest(X_train, y_train)
    rf_results = trainer.evaluate_model(rf_model, X_val, y_val, 'random_forest')
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Save models (to project root models folder)
    trainer.save_models(output_dir='models')
    
    # Final summary
    print("\n" + "="*70)
    print("✅ BASELINE TRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Next Steps:")
    print("   1. Train advanced models (XGBoost, LightGBM)")
    print("   2. Tune hyperparameters with Optuna")
    print("   3. Compare all models")
    print("="*70)


if __name__ == "__main__":
    main()