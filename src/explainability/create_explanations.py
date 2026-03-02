import pandas as pd
import numpy as np
import joblib
import os


class ExplanationGenerator:
    """Generate human-readable explanations from SHAP values"""
    
    def __init__(self):
        """Initialize explanation generator"""
        self.shap_data = None
        self.feature_stats = None
        
    def load_shap_data(self, shap_path='models/shap_values.pkl'):
        """Load pre-computed SHAP values"""
        print("="*70)
        print("LOADING SHAP DATA")
        print("="*70)
        
        print(f"\n📦 Loading SHAP values from: {shap_path}")
        self.shap_data = joblib.load(shap_path)
        print("✅ SHAP data loaded!")
        print(f"   Samples: {self.shap_data['shap_values'].shape[0]}")
        print(f"   Features: {len(self.shap_data['feature_names'])}")
        
        return self.shap_data
    
    def calculate_feature_stats(self, data_path='data/processed/train_features.csv'):
        """Calculate baseline statistics for features"""
        print("\n" + "="*70)
        print("CALCULATING BASELINE STATISTICS")
        print("="*70)
        
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Get feature columns
        exclude_cols = ['machine_id', 'timestamp', 'hour_of_operation', 'failure', 'hours_to_failure']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Calculate statistics
        stats = {}
        for col in feature_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        self.feature_stats = stats
        print(f"✅ Statistics calculated for {len(stats)} features")
        
        return stats
    
    def explain_prediction(self, prediction_idx, top_n=5):
        """Generate explanation for a single prediction"""
        
        # Get SHAP values for this prediction
        shap_vals = self.shap_data['shap_values'][prediction_idx]
        feature_vals = self.shap_data['X_sample'][prediction_idx]
        feature_names = self.shap_data['feature_names']
        base_value = self.shap_data['base_value']
        
        # Handle base_value (might be array or scalar)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value.item()) if base_value.size == 1 else float(base_value.flat[0])
        else:
            base_value = float(base_value)
        
        # Get top contributing features (by absolute SHAP value)
        abs_shap = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap)[::-1][:top_n]
        
        # Build explanation
        explanation = {
            'prediction_id': prediction_idx + 1,
            'base_prediction': base_value,
            'final_prediction': float(base_value + shap_vals.sum()),
            'top_factors': [],
            'text_explanation': ""
        }
        
        factors = []
        for idx in top_indices:
            feature = feature_names[idx]
            value = float(feature_vals[idx])
            shap_value = float(shap_vals[idx])
            impact = "increases" if shap_value > 0 else "decreases"
            
            # Get baseline stats
            if self.feature_stats and feature in self.feature_stats:
                baseline_mean = self.feature_stats[feature]['mean']
                baseline_std = self.feature_stats[feature]['std']
                z_score = (value - baseline_mean) / baseline_std if baseline_std > 0 else 0
                
                # Describe deviation
                if abs(z_score) > 2:
                    deviation = "significantly exceeds" if z_score > 0 else "significantly below"
                elif abs(z_score) > 1:
                    deviation = "exceeds" if z_score > 0 else "below"
                else:
                    deviation = "near"
                
                deviation_desc = f"{deviation} baseline ({baseline_mean:.2f})"
            else:
                deviation_desc = "value recorded"
            
            factor = {
                'feature': feature,
                'value': value,
                'shap_value': shap_value,
                'impact': impact,
                'deviation': deviation_desc
            }
            factors.append(factor)
        
        explanation['top_factors'] = factors
        
        # Generate text explanation
        text_parts = []
        for i, factor in enumerate(factors[:3]):  # Top 3 for text
            feature_clean = factor['feature'].replace('_', ' ')
            text = f"{feature_clean} = {factor['value']:.2f} ({factor['deviation']}) {factor['impact']} failure risk"
            text_parts.append(text)
        
        explanation['text_explanation'] = "; ".join(text_parts)
        
        return explanation
    
    def generate_all_explanations(self, output_dir='reports/week3/explanations', sample_size=10):
        """Generate explanations for multiple predictions"""
        print("\n" + "="*70)
        print("GENERATING EXPLANATIONS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        n_samples = min(sample_size, len(self.shap_data['shap_values']))
        print(f"\n🔄 Generating {n_samples} explanations...")
        
        all_explanations = []
        
        for i in range(n_samples):
            explanation = self.explain_prediction(i, top_n=5)
            all_explanations.append(explanation)
            
            # Save individual explanation
            with open(f'{output_dir}/prediction_{i+1}_explanation.txt', 'w') as f:
                f.write(f"PREDICTION EXPLANATION #{i+1}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Final Prediction Score: {explanation['final_prediction']:.4f}\n")
                f.write(f"(Base risk: {explanation['base_prediction']:.4f})\n\n")
                
                f.write("TOP CONTRIBUTING FACTORS:\n")
                f.write("-"*60 + "\n")
                for j, factor in enumerate(explanation['top_factors'], 1):
                    f.write(f"\n{j}. {factor['feature']}\n")
                    f.write(f"   Value: {factor['value']:.4f}\n")
                    f.write(f"   Deviation: {factor['deviation']}\n")
                    f.write(f"   Impact: {factor['impact']} failure risk\n")
                    f.write(f"   SHAP value: {factor['shap_value']:+.4f}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("HUMAN-READABLE EXPLANATION:\n")
                f.write("-"*60 + "\n")
                f.write(f"{explanation['text_explanation']}\n")
        
        print(f"✅ Generated {n_samples} explanations")
        print(f"✅ Saved to: {output_dir}/")
        
        return all_explanations
    
    def create_summary_report(self, explanations, output_path='reports/week3/explanation_summary.txt'):
        """Create summary report of all explanations"""
        print("\n" + "="*70)
        print("CREATING SUMMARY REPORT")
        print("="*70)
        
        with open(output_path, 'w') as f:
            f.write("SHAP EXPLANATION SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Predictions Analyzed: {len(explanations)}\n\n")
            
            # Aggregate feature importance
            feature_counts = {}
            for exp in explanations:
                for factor in exp['top_factors']:
                    feat = factor['feature']
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1
            
            f.write("MOST FREQUENTLY IMPORTANT FEATURES:\n")
            f.write("-"*70 + "\n")
            sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (feat, count) in enumerate(sorted_features[:10], 1):
                f.write(f"{i:2d}. {feat:50s} (appeared in {count}/{len(explanations)} predictions)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("SAMPLE EXPLANATIONS:\n")
            f.write("-"*70 + "\n\n")
            
            for i, exp in enumerate(explanations[:5], 1):
                f.write(f"{i}. {exp['text_explanation']}\n\n")
        
        print(f"✅ Summary report saved: {output_path}")


def main():
    """Main execution"""
    
    print("="*70)
    print("HUMAN-READABLE EXPLANATIONS - WEEK 3")
    print("="*70)
    
    # Initialize generator
    generator = ExplanationGenerator()
    
    # Load SHAP data
    shap_path = 'models/shap_values.pkl'
    if not os.path.exists(shap_path):
        print(f"❌ ERROR: SHAP values not found at {shap_path}")
        print("   Please run generate_shap.py first!")
        return
    
    generator.load_shap_data(shap_path)
    
    # Calculate baseline statistics
    generator.calculate_feature_stats()
    
    # Generate explanations
    explanations = generator.generate_all_explanations(sample_size=10)
    
    # Create summary report
    generator.create_summary_report(explanations)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EXPLANATION GENERATION COMPLETE!")
    print("="*70)
    print("\n📁 Files Created:")
    print("   ✅ reports/week3/explanations/ (10 individual explanations)")
    print("   ✅ reports/week3/explanation_summary.txt")
    print("\n💡 Example Explanation:")
    if explanations:
        print(f"   {explanations[0]['text_explanation']}")
    print("="*70)


if __name__ == "__main__":
    main()