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
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class ImbalanceHandler:
    """Handle class imbalance using SMOTE"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.smote_applied = False
        
    def load_data(self, train_path):
        """Load training data"""
        print("="*70)
        print("LOADING DATA FOR IMBALANCE HANDLING")
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
        
        # Show class distribution
        print(f"\n📊 Original Training Class Distribution:")
        print(f"   Normal (0): {(y_train==0).sum():,} ({(y_train==0).mean()*100:.4f}%)")
        print(f"   Failure (1): {(y_train==1).sum():,} ({(y_train==1).mean()*100:.4f}%)")
        print(f"   Imbalance Ratio: {(y_train==0).sum() / (y_train==1).sum():.1f}:1")
        
        return X_train, X_val, y_train, y_val
    
    def apply_smote(self, X_train, y_train, sampling_strategy='auto'):
        """Apply SMOTE oversampling"""
        print("\n" + "="*70)
        print("APPLYING SMOTE OVERSAMPLING")
        print("="*70)
        
        # Check if we have enough minority samples
        minority_count = (y_train == 1).sum()
        print(f"\n📊 Minority class count: {minority_count}")
        
        if minority_count < 2:
            print("❌ ERROR: Need at least 2 failure samples for SMOTE")
            print("   Your dataset has too few failures to apply SMOTE effectively.")
            return X_train, y_train
        
        # Set k_neighbors based on minority count
        # SMOTE needs at least k_neighbors+1 samples
        k_neighbors = min(minority_count - 1, 1)
        
        print(f"\n🔄 Applying SMOTE with sampling_strategy='{sampling_strategy}'...")
        print(f"   k_neighbors={k_neighbors} (based on {minority_count} minority samples)")
        print(f"⏱️  This may take a few minutes...")
        
        # Initialize SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=k_neighbors
        )
        
        # Apply SMOTE
        try:
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            self.smote_applied = True
        except Exception as e:
            print(f"\n⚠️  SMOTE failed: {e}")
            print("   Returning original data without SMOTE.")
            return X_train, y_train
        
        self.smote_applied = True
        
        # Show new distribution
        print(f"\n✅ SMOTE Applied Successfully!")
        print(f"\n📊 After SMOTE Class Distribution:")
        print(f"   Normal (0): {(y_train_smote==0).sum():,} ({(y_train_smote==0).mean()*100:.2f}%)")
        print(f"   Failure (1): {(y_train_smote==1).sum():,} ({(y_train_smote==1).mean()*100:.2f}%)")
        print(f"   New Imbalance Ratio: {(y_train_smote==0).sum() / (y_train_smote==1).sum():.1f}:1")
        print(f"\n📈 Training Set Size:")
        print(f"   Before SMOTE: {len(X_train):,} samples")
        print(f"   After SMOTE: {len(X_train_smote):,} samples")
        print(f"   Increase: {len(X_train_smote) - len(X_train):,} samples (+{(len(X_train_smote)/len(X_train)-1)*100:.1f}%)")
        
        return X_train_smote, y_train_smote
    
    def train_xgboost_with_smote(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with SMOTE-balanced data"""
        print("\n" + "="*70)
        print("TRAINING XGBOOST WITH SMOTE")
        print("="*70)
        
        # No need for scale_pos_weight since data is balanced
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='aucpr'
        )
        
        print("\n🔄 Training XGBoost on SMOTE-balanced data...")
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10
        )
        print("✅ Training complete!")
        
        self.models['xgboost_smote'] = xgb_model
        return xgb_model
    
    def train_lightgbm_with_smote(self, X_train, y_train, X_val, y_val):
        """Train LightGBM with SMOTE-balanced data"""
        print("\n" + "="*70)
        print("TRAINING LIGHTGBM WITH SMOTE")
        print("="*70)
        
        # No need for scale_pos_weight since data is balanced
        lgb_model = lgb.LGBMClassifier(
            n_estimators=250,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.75,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        print("\n🔄 Training LightGBM on SMOTE-balanced data...")
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='average_precision',
            callbacks=[lgb.log_evaluation(period=10)]
        )
        print("✅ Training complete!")
        
        self.models['lightgbm_smote'] = lgb_model
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
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Store results
        self.results[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        # Print results
        print(f"\n📊 Performance Metrics:")
        print(f"   PR-AUC:    {pr_auc:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"   TN: {tn:6,} | FP: {fp:6,}")
        print(f"   FN: {fn:6,} | TP: {tp:6,}")
        
        # False alarm rate
        if (fp + tn) > 0:
            false_alarm_rate = fp / (fp + tn)
            print(f"\n⚠️  False Alarm Rate: {false_alarm_rate:.4f} ({false_alarm_rate*100:.2f}%)")
        
        return self.results[model_name]
    
    def compare_with_without_smote(self, models_dir='models'):
        """Compare SMOTE models with original models"""
        print("\n" + "="*70)
        print("COMPARISON: WITH vs WITHOUT SMOTE")
        print("="*70)
        
        # Load original models (without SMOTE)
        try:
            xgb_original = joblib.load(f'{models_dir}/xgboost_tuned.pkl')
            lgb_original = joblib.load(f'{models_dir}/lightgbm_tuned.pkl')
            print("✅ Loaded original models (without SMOTE)")
        except:
            print("⚠️  Original models not found. Showing SMOTE results only.")
            return
        
        # Create comparison table
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name + ' (with SMOTE)',
                'PR-AUC': results['pr_auc'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'False Positives': results['false_positives']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n📊 Performance with SMOTE:")
        print(comparison_df.to_string(index=False))
        
        print("\n💡 Key Observations:")
        print("   • SMOTE increases minority class samples")
        print("   • May improve recall (catch more failures)")
        print("   • May reduce precision (more false alarms)")
        print("   • Trade-off depends on business cost function")
    
    def save_models(self, output_dir='models'):
        """Save SMOTE-trained models"""
        print("\n" + "="*70)
        print("SAVING SMOTE MODELS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{model_name}.pkl"
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"✅ Saved: {filepath}")
        
        # Save results
        results_path = os.path.join(output_dir, 'smote_results.pkl')
        joblib.dump(self.results, results_path)
        print(f"✅ Saved: {results_path}")


def main():
    """Main execution"""
    
    print("="*70)
    print("IMBALANCE HANDLING WITH SMOTE - WEEK 2")
    print("="*70)
    
    # Initialize handler
    handler = ImbalanceHandler()
    
    # Load data
    train_path = 'data/processed/train_features.csv'
    
    if not os.path.exists(train_path):
        print(f"❌ ERROR: Cannot find {train_path}")
        return
    
    print(f"✅ Found data file: {train_path}")
    train_df = handler.load_data(train_path)
    
    # Split data
    X_train, X_val, y_train, y_val = handler.split_data(train_df, test_size=0.2)
    
    # Apply SMOTE
    X_train_smote, y_train_smote = handler.apply_smote(X_train, y_train, sampling_strategy='auto')
    
    # Train XGBoost with SMOTE
    xgb_model = handler.train_xgboost_with_smote(X_train_smote, y_train_smote, X_val, y_val)
    handler.evaluate_model(xgb_model, X_val, y_val, 'xgboost_smote')
    
    # Train LightGBM with SMOTE
    lgb_model = handler.train_lightgbm_with_smote(X_train_smote, y_train_smote, X_val, y_val)
    handler.evaluate_model(lgb_model, X_val, y_val, 'lightgbm_smote')
    
    # Compare with original models
    handler.compare_with_without_smote()
    
    # Save models
    handler.save_models()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ SMOTE IMBALANCE HANDLING COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"   XGBoost (SMOTE) PR-AUC: {handler.results['xgboost_smote']['pr_auc']:.4f}")
    print(f"   LightGBM (SMOTE) PR-AUC: {handler.results['lightgbm_smote']['pr_auc']:.4f}")
    print("\n📁 Files Saved:")
    print("   ✅ models/xgboost_smote.pkl")
    print("   ✅ models/lightgbm_smote.pkl")
    print("   ✅ models/smote_results.pkl")
    print("\n💡 Recommendation:")
    print("   Compare SMOTE vs non-SMOTE models based on:")
    print("   • PR-AUC (primary metric)")
    print("   • Precision (avoid false alarms)")
    print("   • Business cost of false positives vs false negatives")
    print("="*70)


if __name__ == "__main__":
    main()