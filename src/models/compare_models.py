import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """Compare all trained models and select the best"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_models(self, models_dir='models'):
        """Load all trained models"""
        print("="*70)
        print("LOADING ALL TRAINED MODELS")
        print("="*70)
        
        # Model files to load
        model_files = {
            'Logistic Regression': 'logistic_regression_baseline.pkl',
            'Random Forest': 'random_forest_baseline.pkl',
            'XGBoost': 'xgboost_advanced.pkl',
            'LightGBM': 'lightgbm_advanced.pkl',
            'XGBoost (Tuned)': 'xgboost_tuned.pkl',
            'LightGBM (Tuned)': 'lightgbm_tuned.pkl'
        }
        
        loaded_count = 0
        for model_name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"✅ Loaded: {model_name}")
                loaded_count += 1
            else:
                print(f"⚠️  Not found: {model_name} ({filename})")
        
        print(f"\n✅ Loaded {loaded_count}/{len(model_files)} models")
        return loaded_count
    
    def load_data(self, train_path):
        """Load and prepare data"""
        print("\n" + "="*70)
        print("LOADING TEST DATA")
        print("="*70)
        
        print(f"\nLoading data from: {train_path}")
        df = pd.read_csv(train_path)
        print(f"✅ Data loaded: {df.shape}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and target"""
        exclude_cols = [
            'machine_id', 'timestamp', 'hour_of_operation',
            'failure', 'hours_to_failure'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['failure'].copy()
        
        # Handle NaN and inf
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        return X, y
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data by machine_id"""
        unique_machines = df['machine_id'].unique()
        train_machines, test_machines = train_test_split(
            unique_machines, test_size=test_size, random_state=random_state
        )
        
        test_df = df[df['machine_id'].isin(test_machines)]
        
        print(f"\n✅ Test machines: {len(test_machines)}")
        print(f"✅ Test records: {len(test_df):,}")
        
        X_test, y_test = self.prepare_features(test_df)
        
        return X_test, y_test
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS")
        print("="*70)
        
        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # PR-AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
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
                'false_negatives': fn,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"   ✅ PR-AUC: {pr_auc:.4f}")
    
    def create_comparison_table(self):
        """Create comparison table"""
        print("\n" + "="*70)
        print("MODEL COMPARISON TABLE")
        print("="*70)
        
        comparison_df = pd.DataFrame([
            {
                'Model': model_name,
                'PR-AUC': results['pr_auc'],
                'ROC-AUC': results['roc_auc'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'TP': results['true_positives'],
                'FP': results['false_positives'],
                'TN': results['true_negatives'],
                'FN': results['false_negatives']
            }
            for model_name, results in self.results.items()
        ])
        
        # Sort by PR-AUC (most important for imbalanced data)
        comparison_df = comparison_df.sort_values('PR-AUC', ascending=False)
        
        print("\n📊 Performance Comparison (sorted by PR-AUC):")
        print(comparison_df[['Model', 'PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
        
        print("\n📊 Confusion Matrix Details:")
        print(comparison_df[['Model', 'TP', 'FP', 'TN', 'FN']].to_string(index=False))
        
        # Best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_pr_auc = comparison_df.iloc[0]['PR-AUC']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   PR-AUC: {best_pr_auc:.4f}")
        
        return comparison_df, best_model_name
    
    def plot_comparison(self, comparison_df, output_dir='reports/figures/model_comparison'):
        """Create comparison visualizations"""
        print("\n" + "="*70)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # Figure 1: Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['PR-AUC', 'ROC-AUC', 'Precision', 'Recall']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            
            data = comparison_df.sort_values(metric, ascending=True)
            ax.barh(data['Model'], data[metric], color=color, alpha=0.7, edgecolor='black')
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add values on bars
            for i, v in enumerate(data[metric]):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/metrics_comparison.png")
        plt.close()
        
        # Figure 2: PR-AUC Ranking
        fig, ax = plt.subplots(figsize=(10, 6))
        data = comparison_df.sort_values('PR-AUC', ascending=True)
        bars = ax.barh(data['Model'], data['PR-AUC'], 
                      color=['#2ecc71' if i == len(data)-1 else '#3498db' for i in range(len(data))],
                      alpha=0.7, edgecolor='black')
        ax.set_xlabel('PR-AUC Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Ranking by PR-AUC (Primary Metric)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values
        for i, v in enumerate(data['PR-AUC']):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pr_auc_ranking.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/pr_auc_ranking.png")
        plt.close()
    
    def save_best_model(self, best_model_name, output_dir='models'):
        """Copy best model to production file"""
        print("\n" + "="*70)
        print("SAVING BEST MODEL FOR PRODUCTION")
        print("="*70)
        
        best_model = self.models[best_model_name]
        
        # Save as production model
        prod_path = os.path.join(output_dir, 'best_model_production.pkl')
        joblib.dump(best_model, prod_path)
        print(f"✅ Saved best model: {prod_path}")
        
        # Save model info
        info_path = os.path.join(output_dir, 'best_model_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"BEST MODEL FOR PRODUCTION\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Model: {best_model_name}\n")
            f.write(f"PR-AUC: {self.results[best_model_name]['pr_auc']:.4f}\n")
            f.write(f"ROC-AUC: {self.results[best_model_name]['roc_auc']:.4f}\n")
            f.write(f"Precision: {self.results[best_model_name]['precision']:.4f}\n")
            f.write(f"Recall: {self.results[best_model_name]['recall']:.4f}\n")
            f.write(f"F1-Score: {self.results[best_model_name]['f1_score']:.4f}\n")
        
        print(f"✅ Saved model info: {info_path}")
    
    def generate_report(self, comparison_df, output_dir='reports/week2'):
        """Generate final comparison report"""
        print("\n" + "="*70)
        print("GENERATING FINAL REPORT")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'model_comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("MODEL COMPARISON REPORT - WEEK 2\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MODELS EVALUATED:\n")
            f.write("-" * 70 + "\n")
            for i, model_name in enumerate(comparison_df['Model'], 1):
                f.write(f"{i}. {model_name}\n")
            
            f.write("\n\nPERFORMANCE COMPARISON:\n")
            f.write("-" * 70 + "\n")
            f.write(comparison_df.to_string(index=False))
            
            f.write("\n\n\nBEST MODEL:\n")
            f.write("-" * 70 + "\n")
            best = comparison_df.iloc[0]
            f.write(f"Model: {best['Model']}\n")
            f.write(f"PR-AUC: {best['PR-AUC']:.4f}\n")
            f.write(f"ROC-AUC: {best['ROC-AUC']:.4f}\n")
            f.write(f"Precision: {best['Precision']:.4f}\n")
            f.write(f"Recall: {best['Recall']:.4f}\n")
            f.write(f"F1-Score: {best['F1-Score']:.4f}\n")
            
            f.write("\n\nRECOMMENDATION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"The {best['Model']} model is recommended for production deployment.\n")
            f.write(f"It achieved the highest PR-AUC score of {best['PR-AUC']:.4f}, making it\n")
            f.write(f"the most suitable for this imbalanced predictive maintenance task.\n")
        
        print(f"✅ Saved report: {report_path}")
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'model_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV: {csv_path}")


def main():
    """Main execution"""
    
    print("="*70)
    print("MODEL COMPARISON & SELECTION - WEEK 2 FINAL")
    print("="*70)
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load all models
    loaded = comparator.load_models()
    
    if loaded == 0:
        print("\n❌ No models found! Please train models first.")
        return
    
    # Load and prepare data
    train_path = 'data/processed/train_features.csv'
    if not os.path.exists(train_path):
        print(f"❌ Cannot find {train_path}")
        return
    
    df = comparator.load_data(train_path)
    X_test, y_test = comparator.split_data(df, test_size=0.2)
    
    # Evaluate all models
    comparator.evaluate_all_models(X_test, y_test)
    
    # Create comparison table
    comparison_df, best_model_name = comparator.create_comparison_table()
    
    # Create visualizations
    comparator.plot_comparison(comparison_df)
    
    # Save best model
    comparator.save_best_model(best_model_name)
    
    # Generate report
    comparator.generate_report(comparison_df)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ MODEL COMPARISON COMPLETE!")
    print("="*70)
    print(f"\n🏆 Winner: {best_model_name}")
    print(f"\n📁 Files Created:")
    print(f"   ✅ models/best_model_production.pkl")
    print(f"   ✅ reports/week2/model_comparison_report.txt")
    print(f"   ✅ reports/figures/model_comparison/ (visualizations)")
    print("\n🎉 WEEK 2 COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()