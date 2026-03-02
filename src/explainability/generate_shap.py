import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """Generate SHAP values and visualizations"""
    
    def __init__(self, model_path):
        """Initialize with trained model"""
        print("="*70)
        print("SHAP ANALYSIS - MODEL EXPLAINABILITY")
        print("="*70)
        
        print(f"\n📦 Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        
        self.shap_values = None
        self.explainer = None
        self.feature_names = None
        self.base_value = None
        
    def load_data(self, data_path, sample_size=1000):
        """Load and prepare data for SHAP analysis"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape}")
        
        # Prepare features
        exclude_cols = [
            'machine_id', 'timestamp', 'hour_of_operation',
            'failure', 'hours_to_failure'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        self.feature_names = feature_cols
        
        # Sample data for SHAP (SHAP is computationally expensive)
        if len(X) > sample_size:
            print(f"\n📊 Sampling {sample_size} records for SHAP analysis...")
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
            
        print(f"✅ Sample prepared: {X_sample.shape}")
        print(f"✅ Features: {len(feature_cols)}")
        
        return X_sample, X
    
    def calculate_shap_values(self, X_sample, X_background=None):
        """Calculate SHAP values using TreeExplainer"""
        print("\n" + "="*70)
        print("CALCULATING SHAP VALUES")
        print("="*70)
        
        print("\n🔄 Creating SHAP explainer...")
        print("⏱️  This may take 5-10 minutes...")
        
        # Create TreeExplainer (works with XGBoost, LightGBM, etc.)
        self.explainer = shap.TreeExplainer(self.model)
        
        print("✅ Explainer created!")
        print(f"   Expected value: {self.explainer.expected_value}")
        print("\n🔄 Computing SHAP values...")
        
        # Calculate SHAP values
        shap_values_raw = self.explainer.shap_values(X_sample)
        
        # Handle different output formats
        if isinstance(shap_values_raw, list):
            # Binary classification - use positive class
            self.shap_values = shap_values_raw[1]
            self.base_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
        elif len(shap_values_raw.shape) == 3:
            # 3D array (samples, features, classes)
            self.shap_values = shap_values_raw[:, :, 1]
            self.base_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
        else:
            # Already 2D array
            self.shap_values = shap_values_raw
            self.base_value = self.explainer.expected_value
        
        print("✅ SHAP values computed!")
        print(f"   Shape: {self.shap_values.shape}")
        print(f"   Base value: {self.base_value}")
        
        return self.shap_values
    
    def create_summary_plot(self, X_sample, output_dir='reports/figures/shap'):
        """Create SHAP summary plot"""
        print("\n" + "="*70)
        print("CREATING SHAP SUMMARY PLOT")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Summary plot (bar chart of mean absolute SHAP values)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X_sample,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False,
            max_display=20
        )
        plt.title('SHAP Feature Importance (Mean |SHAP value|)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/shap_summary_bar.png")
        
        # Summary plot (beeswarm - shows distribution)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=20
        )
        plt.title('SHAP Summary Plot (Feature Impact Distribution)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/shap_summary_beeswarm.png")
    
    def create_waterfall_plots(self, X_sample, output_dir='reports/figures/shap/waterfall', n_samples=5):
        """Create waterfall plots for individual predictions"""
        print("\n" + "="*70)
        print("CREATING WATERFALL PLOTS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🔄 Creating {n_samples} waterfall plots...")
        
        success_count = 0
        for i in range(min(n_samples, len(X_sample))):
            try:
                # Get single prediction SHAP values
                single_shap = self.shap_values[i]
                
                # Make sure we have 1D array
                if len(single_shap.shape) > 1:
                    single_shap = single_shap.flatten()
                
                # Create explanation object for waterfall plot
                explanation = shap.Explanation(
                    values=single_shap,
                    base_values=float(self.base_value) if isinstance(self.base_value, (np.ndarray, np.generic)) else self.base_value,
                    data=X_sample.iloc[i].values,
                    feature_names=list(self.feature_names)
                )
                
                # Create figure
                fig = plt.figure(figsize=(10, 6))
                
                # Create waterfall plot
                shap.plots.waterfall(explanation, show=False, max_display=15)
                
                plt.title(f'SHAP Waterfall Plot - Prediction {i+1}', fontsize=12, fontweight='bold', pad=20)
                plt.tight_layout()
                
                # Save
                save_path = f'{output_dir}/waterfall_prediction_{i+1}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                success_count += 1
                print(f"   ✅ Created waterfall plot {i+1}")
                
            except Exception as e:
                print(f"   ⚠️  Skipped waterfall plot {i+1}: {str(e)[:50]}")
                plt.close('all')
                continue
            
        if success_count > 0:
            print(f"\n✅ Saved {success_count}/{n_samples} waterfall plots to {output_dir}/")
        else:
            print(f"\n⚠️  No waterfall plots created. Creating alternative force plots...")
            self.create_force_plots_alternative(X_sample, output_dir, n_samples)
    
    def create_force_plots_alternative(self, X_sample, output_dir, n_samples=5):
        """Create force plots as alternative to waterfall plots"""
        print(f"   Creating force plots instead...")
        
        for i in range(min(n_samples, len(X_sample))):
            try:
                # Create simple bar plot showing top features
                single_shap = self.shap_values[i]
                
                # Get top 10 features
                abs_vals = np.abs(single_shap)
                top_indices = np.argsort(abs_vals)[::-1][:10]
                
                # Create bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                top_features = [self.feature_names[idx] for idx in top_indices]
                top_shap_vals = single_shap[top_indices]
                
                colors = ['red' if x > 0 else 'blue' for x in top_shap_vals]
                ax.barh(range(len(top_features)), top_shap_vals, color=colors, alpha=0.7)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features)
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title(f'Feature Contribution - Prediction {i+1}')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/feature_contrib_prediction_{i+1}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"   ⚠️  Also failed alternative plot {i+1}")
                plt.close('all')
                continue
    
    def create_dependence_plots(self, X_sample, output_dir='reports/figures/shap/dependence', top_n=5):
        """Create dependence plots for top features"""
        print("\n" + "="*70)
        print("CREATING DEPENDENCE PLOTS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get top features by mean absolute SHAP value
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[::-1][:top_n]
        
        print(f"\n🔄 Creating dependence plots for top {top_n} features...")
        
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                idx,
                self.shap_values,
                X_sample,
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f'SHAP Dependence Plot: {feature_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # Clean filename
            clean_name = feature_name.replace('/', '_').replace(' ', '_')
            plt.savefig(f'{output_dir}/dependence_{clean_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"✅ Saved {top_n} dependence plots to {output_dir}/")
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from SHAP values"""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE FROM SHAP")
        print("="*70)
        
        # Calculate mean absolute SHAP value for each feature
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def save_shap_values(self, X_sample, output_dir='models'):
        """Save SHAP values for later use"""
        print("\n" + "="*70)
        print("SAVING SHAP VALUES")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save SHAP values
        shap_data = {
            'shap_values': self.shap_values,
            'feature_names': self.feature_names,
            'base_value': self.base_value,
            'X_sample': X_sample.values
        }
        
        shap_path = os.path.join(output_dir, 'shap_values.pkl')
        joblib.dump(shap_data, shap_path)
        print(f"✅ Saved: {shap_path}")
        
        # Save feature importance
        importance_df = self.get_feature_importance(top_n=len(self.feature_names))
        importance_path = os.path.join(output_dir, 'shap_feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"✅ Saved: {importance_path}")


def main():
    """Main execution"""
    
    print("="*70)
    print("SHAP EXPLAINABILITY ANALYSIS - WEEK 3")
    print("="*70)
    
    # Paths
    model_path = 'models/best_model_production.pkl'
    data_path = 'data/processed/train_features.csv'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        print("   Please train models first (Week 2)")
        return
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data not found at {data_path}")
        return
    
    # Initialize analyzer
    analyzer = SHAPAnalyzer(model_path)
    
    # Load data
    X_sample, X_full = analyzer.load_data(data_path, sample_size=1000)
    
    # Calculate SHAP values
    shap_values = analyzer.calculate_shap_values(X_sample, X_full)
    
    # Create visualizations
    analyzer.create_summary_plot(X_sample)
    analyzer.create_waterfall_plots(X_sample, n_samples=5)
    analyzer.create_dependence_plots(X_sample, top_n=5)
    
    # Get feature importance
    importance_df = analyzer.get_feature_importance(top_n=20)
    
    # Save SHAP values
    analyzer.save_shap_values(X_sample)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ SHAP ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📊 Visualizations Created:")
    print("   ✅ SHAP summary plots (2)")
    print("   ✅ Waterfall plots (5)")
    print("   ✅ Dependence plots (5)")
    print("\n📁 Files Saved:")
    print("   ✅ models/shap_values.pkl")
    print("   ✅ models/shap_feature_importance.csv")
    print("\n🎯 Next Steps:")
    print("   1. Review SHAP visualizations")
    print("   2. Generate human-readable explanations")
    print("   3. Create explanation dashboard")
    print("="*70)


if __name__ == "__main__":
    main()