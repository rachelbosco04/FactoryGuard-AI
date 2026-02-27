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
import os

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

        exclude = [
            'machine_id',
            'timestamp',
            'hour_of_operation',
            'failure',
            'hours_to_failure'
        ]

        sensor_cols = [
            col for col in df.columns
            if col not in exclude and df[col].dtype in ['float64', 'int64']
        ]

        self.sensor_columns = sensor_cols
        return sensor_cols

    def handle_missing_values(self, df, method='forward_fill'):
        """Handle missing values"""

        print("\n🔍 Handling Missing Values...")

        missing_before = df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")

        if missing_before == 0:
            print("✅ No missing values found!")
            return df

        df_clean = df.copy()

        if method == 'forward_fill':
            df_clean = df_clean.groupby('machine_id').ffill()
            df_clean = df_clean.groupby('machine_id').bfill()

        elif method == 'interpolate':
            for col in self.sensor_columns:
                df_clean[col] = df_clean.groupby('machine_id')[col].transform(
                    lambda x: x.interpolate(limit_direction='both')
                )

        elif method == 'mean':
            df_clean[self.sensor_columns] = df_clean[self.sensor_columns].fillna(
                df_clean[self.sensor_columns].mean()
            )

        elif method == 'drop':
            df_clean = df_clean.dropna()

        missing_after = df_clean.isnull().sum().sum()

        print(f"Missing values after: {missing_after}")
        print(f"✅ Fixed {missing_before - missing_after} missing values")

        self.cleaning_stats['missing_values_removed'] = missing_before - missing_after

        return df_clean

    def remove_outliers(self, df, method='iqr', threshold=3.0):
        """Remove outliers"""

        print("\n🔍 Detecting Outliers...")

        df_clean = df.copy()
        outliers_detected = 0

        for sensor in self.sensor_columns:

            if method == 'iqr':

                Q1 = df_clean[sensor].quantile(0.25)
                Q3 = df_clean[sensor].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                mask = (df_clean[sensor] < lower) | (df_clean[sensor] > upper)
                outliers_detected += mask.sum()

                df_clean.loc[df_clean[sensor] < lower, sensor] = lower
                df_clean.loc[df_clean[sensor] > upper, sensor] = upper

            elif method == 'zscore':

                z = np.abs(stats.zscore(df_clean[sensor]))
                mask = z > threshold

                outliers_detected += mask.sum()

                median = df_clean[sensor].median()
                df_clean.loc[mask, sensor] = median

        print(f"✅ Handled {outliers_detected:,} outliers")

        self.cleaning_stats['outliers_handled'] = outliers_detected

        return df_clean

    def remove_noise(self, df, method='rolling', window=5):
        """Smooth noise"""

        print("\n🔍 Removing Noise...")

        df_clean = df.copy()

        for sensor in self.sensor_columns:

            if method == 'rolling':

                df_clean[sensor] = df_clean.groupby('machine_id')[sensor].transform(
                    lambda x: x.rolling(window=window, center=True,
                                        min_periods=1).mean()
                )

            elif method == 'ewm':

                df_clean[sensor] = df_clean.groupby('machine_id')[sensor].transform(
                    lambda x: x.ewm(span=window).mean()
                )

        print(f"✅ Smoothed {len(self.sensor_columns)} sensors")

        return df_clean

    def remove_constant_features(self, df):
        """Remove constant columns"""

        print("\n🔍 Checking Constants...")

        constant = []

        for col in self.sensor_columns:
            if df[col].nunique() == 1:
                constant.append(col)

        if constant:
            print("Removing:", constant)
            df = df.drop(columns=constant)
            self.sensor_columns = [c for c in self.sensor_columns if c not in constant]

        else:
            print("✅ No constants")

        return df

    def remove_duplicates(self, df):
        """Remove duplicates"""

        print("\n🔍 Checking Duplicates...")

        dup = df.duplicated().sum()

        if dup > 0:
            df = df.drop_duplicates()
            print(f"Removed {dup} duplicates")
        else:
            print("✅ No duplicates")

        self.cleaning_stats['duplicates_removed'] = dup

        return df

    def normalize_features(self, df, method='robust'):
        """Normalize data"""

        print("\n🔍 Normalizing...")

        df_clean = df.copy()

        if method == 'standard':
            self.scaler = StandardScaler()

        elif method == 'robust':
            self.scaler = RobustScaler()

        df_clean[self.sensor_columns] = self.scaler.fit_transform(
            df_clean[self.sensor_columns]
        )

        print(f"✅ Normalized {len(self.sensor_columns)} columns")

        return df_clean

    def clean_and_preprocess(self, df):

        print("=" * 60)
        print("DATA CLEANING PIPELINE")
        print("=" * 60)

        self.identify_sensor_columns(df)

        print(f"Sensors: {len(self.sensor_columns)}")

        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.remove_constant_features(df)
        df = self.remove_outliers(df)
        df = self.remove_noise(df)

        print("=" * 60)
        print("DONE CLEANING")
        print("=" * 60)

        return df

    def save_cleaned_data(self, df, filename):

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )

        output_dir = os.path.join(base_dir, "data", "processed")

        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, filename)

        df.to_csv(path, index=False)

        print(f"\n💾 Saved to {path}")


def main():

    print("Loading raw data...")

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    train_path = os.path.join(base_dir, "data", "raw", "train_data.csv")
    test_path = os.path.join(base_dir, "data", "raw", "test_data.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    preprocessor = DataPreprocessor()

    print("\n📊 TRAIN DATA")
    train_clean = preprocessor.clean_and_preprocess(train_df)
    preprocessor.save_cleaned_data(train_clean, "train_cleaned.csv")

    print("\n📊 TEST DATA")
    test_clean = preprocessor.clean_and_preprocess(test_df)
    preprocessor.save_cleaned_data(test_clean, "test_cleaned.csv")

    print("\n🎉 COMPLETE")


if __name__ == "__main__":
    main()