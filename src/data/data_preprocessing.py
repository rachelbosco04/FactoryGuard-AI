"""
Data Cleaning and Preprocessing for Predictive Maintenance
==========================================================
Handles missing values, outliers, noise removal, and normalization
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data cleaning and preprocessing pipeline
    """
    
    def __init__(self):
        self.scaler = None
        self.sensor_columns = None
        self.cleaning_stats = {}
        
    def identify_sensor_columns(self, df):
        """Identify sensor and numerical columns"""
        # Exclude ID, timestamp, and target columns
        exclude = ['machine_id', 'timestamp', 'hour_of_operation', 
                   'failure', 'hours_to_failure']
        
        sensor_cols = [col for col in df.columns 
                      if col not in exclude and df[col].dtype in ['float64', 'int64']]
        
        self.sensor_columns = sensor_cols
        return sensor_cols
    
    def handle_missing_values(self, df, method='forward_fill'):
        """
        Handle missing values in sensor data
        
        Methods:
        - forward_fill: Use previous value (good for time series)
        - interpolate: Linear interpolation
        - mean: Fill with column mean
        - drop: Remove rows with missing values
        """
        print("\n🔍 Handling Missing Values...")
        
        missing_before = df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")
        
        if missing_before == 0:
            print("✅ No missing values found!")
            return df
        
        df_clean = df.copy()
        
        if method == 'forward_fill':
            # Forward fill within each machine
            df_clean = df_clean.groupby('machine_id').fillna(method='ffill')
            # Backward fill for any remaining NaN at start
            df_clean = df_clean.groupby('machine_id').fillna(method='bfill')
            
        elif method == 'interpolate':
            # Linear interpolation within each machine
            for col in self.sensor_columns:
                df_clean[col] = df_clean.groupby('machine_id')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
                
        elif method == 'mean':
            # Fill with column mean
            df_clean[self.sensor_columns] = df_clean[self.sensor_columns].fillna(
                df_clean[self.sensor_columns].mean()
            )
            
        elif method == 'drop':
            df_clean = df_clean.dropna()
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"Missing values after: {missing_after}")
        print(f"✅ Removed/filled {missing_before - missing_after} missing values")
        
        self.cleaning_stats['missing_values_removed'] = missing_before - missing_after
        
        return df_clean
    
    def remove_outliers(self, df, method='iqr', threshold=3.0):
        """
        Remove or cap outliers in sensor data
        
        Methods:
        - iqr: Interquartile Range method (recommended for skewed data)
        - zscore: Z-score method (for normal distributions)
        - cap: Cap outliers instead of removing
        """
        print(f"\n🔍 Detecting Outliers (method: {method})...")
        
        df_clean = df.copy()
        outliers_detected = 0
        
        for sensor in self.sensor_columns:
            if method == 'iqr':
                # IQR method (more robust)
                Q1 = df_clean[sensor].quantile(0.25)
                Q3 = df_clean[sensor].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df_clean[sensor] < lower_bound) | (df_clean[sensor] > upper_bound)
                outliers_detected += outlier_mask.sum()
                
                # Cap outliers instead of removing (preserves data)
                df_clean.loc[df_clean[sensor] < lower_bound, sensor] = lower_bound
                df_clean.loc[df_clean[sensor] > upper_bound, sensor] = upper_bound
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(df_clean[sensor]))
                outlier_mask = z_scores > threshold
                outliers_detected += outlier_mask.sum()
                
                # Replace outliers with median
                median_val = df_clean[sensor].median()
                df_clean.loc[outlier_mask, sensor] = median_val
        
        print(f"✅ Detected and handled {outliers_detected:,} outlier values")
        self.cleaning_stats['outliers_handled'] = outliers_detected
        
        return df_clean
    
    def remove_noise(self, df, method='rolling', window=5):
        """
        Remove sensor noise using smoothing techniques
        
        Methods:
        - rolling: Rolling window average (simple smoothing)
        - ewm: Exponential weighted moving average (adaptive)
        - savgol: Savitzky-Golay filter (preserves peaks)
        """
        print(f"\n🔍 Removing Noise (method: {method}, window: {window})...")
        
        df_clean = df.copy()
        
        for sensor in self.sensor_columns:
            if method == 'rolling':
                # Simple rolling average
                df_clean[sensor] = df_clean.groupby('machine_id')[sensor].transform(
                    lambda x: x.rolling(window=window, center=True, min_periods=1).mean()
                )
                
            elif method == 'ewm':
                # Exponential weighted moving average
                df_clean[sensor] = df_clean.groupby('machine_id')[sensor].transform(
                    lambda x: x.ewm(span=window, adjust=False).mean()
                )
                
            elif method == 'savgol':
                # Savitzky-Golay filter (requires scipy)
                from scipy.signal import savgol_filter
                df_clean[sensor] = df_clean.groupby('machine_id')[sensor].transform(
                    lambda x: savgol_filter(x, window_length=window, polyorder=2) 
                    if len(x) >= window else x
                )
        
        print(f"✅ Applied {method} smoothing to {len(self.sensor_columns)} sensors")
        
        return df_clean
    
    def remove_constant_features(self, df):
        """Remove features with zero variance (constant values)"""
        print("\n🔍 Checking for Constant Features...")
        
        constant_features = []
        for col in self.sensor_columns:
            if df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"⚠️ Found {len(constant_features)} constant features: {constant_features}")
            df_clean = df.drop(columns=constant_features)
            self.sensor_columns = [c for c in self.sensor_columns if c not in constant_features]
        else:
            print("✅ No constant features found")
            df_clean = df.copy()
        
        return df_clean
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        print("\n🔍 Checking for Duplicates...")
        
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            print(f"⚠️ Found {duplicates} duplicate rows")
            df_clean = df.drop_duplicates()
            print(f"✅ Removed {duplicates} duplicates")
        else:
            print("✅ No duplicates found")
            df_clean = df.copy()
        
        self.cleaning_stats['duplicates_removed'] = duplicates
        
        return df_clean
    
    def normalize_features(self, df, method='robust'):
        """
        Normalize sensor values
        
        Methods:
        - standard: StandardScaler (mean=0, std=1)
        - robust: RobustScaler (robust to outliers)
        - minmax: Min-Max scaling (0-1 range)
        """
        print(f"\n🔍 Normalizing Features (method: {method})...")
        
        df_clean = df.copy()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        # Fit and transform sensor columns
        df_clean[self.sensor_columns] = self.scaler.fit_transform(
            df_clean[self.sensor_columns]
        )
        
        print(f"✅ Normalized {len(self.sensor_columns)} sensor features")
        
        return df_clean
    
    def clean_and_preprocess(self, df, config=None):
        """
        Complete cleaning and preprocessing pipeline
        
        Args:
            df: Input dataframe
            config: Dictionary with cleaning options
        """
        if config is None:
            config = {
                'handle_missing': True,
                'missing_method': 'forward_fill',
                'remove_outliers': True,
                'outlier_method': 'iqr',
                'remove_noise': True,
                'noise_method': 'rolling',
                'noise_window': 5,
                'remove_duplicates': True,
                'normalize': False,  # Don't normalize before feature engineering
                'normalize_method': 'robust'
            }
        
        print("="*70)
        print("DATA CLEANING & PREPROCESSING PIPELINE")
        print("="*70)
        print(f"Input shape: {df.shape}")
        
        # Identify sensor columns
        self.identify_sensor_columns(df)
        print(f"\nIdentified {len(self.sensor_columns)} sensor columns")
        
        df_clean = df.copy()
        
        # Step 1: Remove duplicates
        if config.get('remove_duplicates'):
            df_clean = self.remove_duplicates(df_clean)
        
        # Step 2: Handle missing values
        if config.get('handle_missing'):
            df_clean = self.handle_missing_values(
                df_clean, 
                method=config.get('missing_method', 'forward_fill')
            )
        
        # Step 3: Remove constant features
        df_clean = self.remove_constant_features(df_clean)
        
        # Step 4: Remove outliers
        if config.get('remove_outliers'):
            df_clean = self.remove_outliers(
                df_clean,
                method=config.get('outlier_method', 'iqr')
            )
        
        # Step 5: Remove noise
        if config.get('remove_noise'):
            df_clean = self.remove_noise(
                df_clean,
                method=config.get('noise_method', 'rolling'),
                window=config.get('noise_window', 5)
            )
        
        # Step 6: Normalize (optional, usually done after feature engineering)
        if config.get('normalize'):
            df_clean = self.normalize_features(
                df_clean,
                method=config.get('normalize_method', 'robust')
            )
        
        print("\n" + "="*70)
        print("CLEANING COMPLETE!")
        print("="*70)
        print(f"Output shape: {df_clean.shape}")
        print(f"Rows removed: {len(df) - len(df_clean):,}")
        print(f"\nCleaning Statistics:")
        for key, value in self.cleaning_stats.items():
            print(f"  - {key}: {value:,}")
        
        return df_clean
    
    def save_cleaned_data(self, df, filename='train_cleaned.csv'):
        """Save cleaned dataset"""
        import os
        os.makedirs('../../data/processed', exist_ok=True)
        
        output_path = f'../../data/processed/{filename}'
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved cleaned data to: {output_path}")


def main():
    """Main execution"""
    
    print("Loading raw data...")
    train_df = pd.read_csv('../../data/raw/train_data.csv')
    test_df = pd.read_csv('../../data/raw/test_data.csv')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Configure cleaning pipeline
    cleaning_config = {
        'handle_missing': True,
        'missing_method': 'forward_fill',
        'remove_outliers': True,
        'outlier_method': 'iqr',
        'remove_noise': True,
        'noise_method': 'rolling',
        'noise_window': 5,
        'remove_duplicates': True,
        'normalize': False,  # We'll normalize after feature engineering
    }
    
    # Clean training data
    print("\n📊 PROCESSING TRAINING DATA")
    train_clean = preprocessor.clean_and_preprocess(train_df, cleaning_config)
    preprocessor.save_cleaned_data(train_clean, 'train_cleaned.csv')
    
    # Clean test data
    print("\n\n📊 PROCESSING TEST DATA")
    test_clean = preprocessor.clean_and_preprocess(test_df, cleaning_config)
    preprocessor.save_cleaned_data(test_clean, 'test_cleaned.csv')
    
    print("\n🎉 All data cleaning complete!")


if __name__ == "__main__":
    main()