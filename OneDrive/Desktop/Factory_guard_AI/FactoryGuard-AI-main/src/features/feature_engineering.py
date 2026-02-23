"""
Advanced Feature Engineering for Predictive Maintenance
=======================================================
Creates time-series features including rolling windows, lags, and degradation indices

Author: Your Name  
Date: February 2026
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Create advanced features for predictive maintenance
    
    Features created:
    - Rolling window statistics (mean, std, min, max)
    - Lag features (previous time steps)
    - Exponential moving averages
    - Rate of change (delta from previous hour)
    - Degradation index (composite health score)
    - Interaction features
    """
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        if config is None:
            # Default configuration
            self.config = {
                'rolling_windows': [6, 12, 24],
                'lag_periods': [1, 2, 3, 6],
                'ema_alphas': [0.1, 0.3],
                'key_sensors': [
                    'spindle_temp', 'spindle_vibration', 'motor_current',
                    'tool_vibration', 'tool_wear', 'hydraulic_pressure',
                    'acoustic_emission'
                ]
            }
        else:
            self.config = config
        
        self.feature_names = []
    
    def identify_sensor_columns(self, df):
        """Identify sensor columns to engineer features from"""
        exclude = ['machine_id', 'timestamp', 'hour_of_operation', 
                   'shift', 'material_type', 'workload_intensity',
                   'failure', 'hours_to_failure']
        
        sensor_cols = [col for col in df.columns if col not in exclude]
        
        # Use only key sensors if specified
        if self.config.get('key_sensors'):
            sensor_cols = [s for s in sensor_cols if s in self.config['key_sensors']]
        
        return sensor_cols
    
    def create_rolling_features(self, df, sensor_cols):
        """Create rolling window statistics"""
        print("🔄 Creating rolling window features...")
        
        df_new = df.copy()
        rolling_windows = self.config['rolling_windows']
        
        for sensor in tqdm(sensor_cols, desc="Rolling features"):
            for window in rolling_windows:
                # Rolling Mean
                df_new[f'{sensor}_rolling_mean_{window}h'] = \
                    df_new.groupby('machine_id')[sensor].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                
                # Rolling Std
                df_new[f'{sensor}_rolling_std_{window}h'] = \
                    df_new.groupby('machine_id')[sensor].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                
                # Rolling Min
                df_new[f'{sensor}_rolling_min_{window}h'] = \
                    df_new.groupby('machine_id')[sensor].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                
                # Rolling Max
                df_new[f'{sensor}_rolling_max_{window}h'] = \
                    df_new.groupby('machine_id')[sensor].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
        
        rolling_cols = [c for c in df_new.columns if 'rolling' in c]
        print(f"✅ Created {len(rolling_cols)} rolling features")
        self.feature_names.extend(rolling_cols)
        
        return df_new
    
    def create_lag_features(self, df, sensor_cols):
        """Create lag features (previous time steps)"""
        print("🔄 Creating lag features...")
        
        df_new = df.copy()
        lag_periods = self.config['lag_periods']
        
        for sensor in tqdm(sensor_cols, desc="Lag features"):
            for lag in lag_periods:
                df_new[f'{sensor}_lag_{lag}h'] = \
                    df_new.groupby('machine_id')[sensor].shift(lag)
        
        # Fill NaN values from lags with backward fill
        lag_cols = [c for c in df_new.columns if 'lag' in c]
        df_new[lag_cols] = df_new.groupby('machine_id')[lag_cols].fillna(method='bfill')
        
        print(f"✅ Created {len(lag_cols)} lag features")
        self.feature_names.extend(lag_cols)
        
        return df_new
    
    def create_ema_features(self, df, sensor_cols):
        """Create exponential moving average features"""
        print("🔄 Creating EMA features...")
        
        df_new = df.copy()
        ema_alphas = self.config['ema_alphas']
        
        for sensor in tqdm(sensor_cols, desc="EMA features"):
            for alpha in ema_alphas:
                df_new[f'{sensor}_ema_{alpha}'] = \
                    df_new.groupby('machine_id')[sensor].transform(
                        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
                    )
        
        ema_cols = [c for c in df_new.columns if 'ema' in c]
        print(f"✅ Created {len(ema_cols)} EMA features")
        self.feature_names.extend(ema_cols)
        
        return df_new
    
    def create_rate_of_change(self, df, sensor_cols):
        """Create rate of change features"""
        print("🔄 Creating rate of change features...")
        
        df_new = df.copy()
        
        for sensor in tqdm(sensor_cols, desc="Rate of change"):
            df_new[f'{sensor}_rate_of_change'] = \
                df_new.groupby('machine_id')[sensor].diff()
        
        # Fill first values with 0
        roc_cols = [c for c in df_new.columns if 'rate_of_change' in c]
        df_new[roc_cols] = df_new[roc_cols].fillna(0)
        
        print(f"✅ Created {len(roc_cols)} rate of change features")
        self.feature_names.extend(roc_cols)
        
        return df_new
    
    def create_degradation_index(self, df, sensor_cols):
        """Create composite degradation index"""
        print("🔄 Creating degradation index...")
        
        df_new = df.copy()
        
        # Normalize sensors to 0-1 scale
        normalized_sensors = []
        for sensor in sensor_cols:
            min_val = df_new[sensor].min()
            max_val = df_new[sensor].max()
            
            if max_val > min_val:
                normalized = (df_new[sensor] - min_val) / (max_val - min_val)
                normalized_sensors.append(normalized)
        
        # Average normalized values
        if normalized_sensors:
            df_new['degradation_index'] = pd.concat(normalized_sensors, axis=1).mean(axis=1)
            print("✅ Created degradation_index")
            self.feature_names.append('degradation_index')
        
        return df_new
    
    def create_interaction_features(self, df):
        """Create interaction features between key sensors"""
        print("🔄 Creating interaction features...")
        
        df_new = df.copy()
        interactions_created = 0
        
        # Temperature × Vibration
        if 'spindle_temp' in df_new.columns and 'spindle_vibration' in df_new.columns:
            df_new['temp_vib_interaction'] = df_new['spindle_temp'] * df_new['spindle_vibration']
            interactions_created += 1
        
        # Power × Current
        if 'power_consumption' in df_new.columns and 'motor_current' in df_new.columns:
            df_new['power_current_interaction'] = df_new['power_consumption'] * df_new['motor_current']
            interactions_created += 1
        
        # Tool wear × Cutting force
        if 'tool_wear' in df_new.columns and 'cutting_force' in df_new.columns:
            df_new['wear_force_interaction'] = df_new['tool_wear'] * df_new['cutting_force']
            interactions_created += 1
        
        # Vibration × Pressure (inverse relationship)
        if 'tool_vibration' in df_new.columns and 'hydraulic_pressure' in df_new.columns:
            df_new['vib_pressure_ratio'] = df_new['tool_vibration'] / (df_new['hydraulic_pressure'] + 1e-6)
            interactions_created += 1
        
        interaction_cols = [c for c in df_new.columns if 'interaction' in c or 'ratio' in c]
        print(f"✅ Created {interactions_created} interaction features")
        self.feature_names.extend(interaction_cols)
        
        return df_new
    
    def engineer_all_features(self, df):
        """Complete feature engineering pipeline"""
        print("\n" + "="*70)
        print("🚀 FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Get sensor columns
        sensor_cols = self.identify_sensor_columns(df)
        
        print(f"\n📊 Input shape: {df.shape}")
        print(f"🔧 Engineering features for {len(sensor_cols)} sensors")
        
        # Reset feature names list
        self.feature_names = []
        
        # Create features step by step
        df_features = df.copy()
        
        df_features = self.create_rolling_features(df_features, sensor_cols)
        df_features = self.create_lag_features(df_features, sensor_cols)
        df_features = self.create_ema_features(df_features, sensor_cols)
        df_features = self.create_rate_of_change(df_features, sensor_cols)
        df_features = self.create_degradation_index(df_features, sensor_cols)
        df_features = self.create_interaction_features(df_features)
        
        # Final statistics
        new_features = len(self.feature_names)
        
        print("\n" + "="*70)
        print("✅ FEATURE ENGINEERING COMPLETE!")
        print("="*70)
        print(f"📊 Output shape: {df_features.shape}")
        print(f"✨ Created {new_features} new features")
        print(f"📈 Total features: {df_features.shape[1]}")
        print("="*70 + "\n")
        
        return df_features
    
    def save_features(self, df, filename):
        """Save engineered features to CSV"""
        import os
        
        # Create directory if it doesn't exist
        output_dir = '../../data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f'{output_dir}/{filename}'
        df.to_csv(output_path, index=False)
        print(f"💾 Saved features to: {output_path}")


def main():
    """Main execution function"""
    
    print("="*70)
    print("PREDICTIVE MAINTENANCE - FEATURE ENGINEERING")
    print("="*70)
    
    # Load cleaned data (if available) or raw data
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

train_cleaned_path = os.path.join(BASE_DIR, 'data', 'processed', 'train_cleaned.csv')
test_cleaned_path = os.path.join(BASE_DIR, 'data', 'processed', 'test_cleaned.csv')

train_raw_path = os.path.join(BASE_DIR, 'data', 'raw', 'train_data.csv')
test_raw_path = os.path.join(BASE_DIR, 'data', 'raw', 'test_data.csv')

try:
    print("\nAttempting to load cleaned data...")
    train_df = pd.read_csv(train_cleaned_path)
    test_df = pd.read_csv(test_cleaned_path)
    print("✅ Loaded cleaned data")
except FileNotFoundError:
    print("⚠️ Cleaned data not found, loading raw data...")
    train_df = pd.read_csv(train_raw_path)
    test_df = pd.read_csv(test_raw_path)
    print("✅ Loaded raw data")
    
    # Initialize feature engineer
    config = {
        'rolling_windows': [6, 12, 24],
        'lag_periods': [1, 2, 3, 6],
        'ema_alphas': [0.1, 0.3],
        'key_sensors': [
            'spindle_temp',
            'spindle_vibration',
            'motor_current',
            'tool_vibration',
            'tool_wear',
            'hydraulic_pressure',
            'acoustic_emission'
        ]
    }
    
    engineer = FeatureEngineer(config)
    
    # Engineer features for training data
    print("\n🎯 PROCESSING TRAINING DATA")
    train_features = engineer.engineer_all_features(train_df)
    engineer.save_features(train_features, 'train_features.csv')
    
    # Engineer features for test data
    print("\n🎯 PROCESSING TEST DATA")
    test_features = engineer.engineer_all_features(test_df)
    engineer.save_features(test_features, 'test_features.csv')
    
    # Summary
    print("\n" + "="*70)
    print("🎉 FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"\n✅ Training features: {train_features.shape}")
    print(f"✅ Test features: {test_features.shape}")
    print(f"\n📁 Files saved:")
    print(f"   ✅ data/processed/train_features.csv")
    print(f"   ✅ data/processed/test_features.csv")
    print("\n🎯 Ready for modeling!")
    print("="*70)


if __name__ == "__main__":
    main()