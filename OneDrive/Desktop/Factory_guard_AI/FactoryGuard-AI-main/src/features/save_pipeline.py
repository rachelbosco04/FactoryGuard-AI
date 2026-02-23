"""
joblib Serialization - Save Feature Engineering Pipeline
"""

import joblib
import pandas as pd
import os
import sys

# Add path to find feature_engineering.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import FeatureEngineer

print("="*60)
print("SAVING PIPELINE WITH JOBLIB")
print("="*60)

# Step 1: Create feature engineer
print("\nStep 1: Creating Feature Engineer...")
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
print("✅ Feature Engineer created!")

# Step 2: Create models folder
print("\nStep 2: Creating models folder...")
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)
print(f"✅ Models folder: {models_dir}")

# Step 3: Save with joblib
print("\nStep 3: Saving with joblib...")
save_path = os.path.join(models_dir, 'feature_engineer.pkl')
joblib.dump(engineer, save_path)
print(f"✅ Saved to: {save_path}")

# Step 4: Save config
config_path = os.path.join(models_dir, 'feature_config.pkl')
joblib.dump(config, config_path)
print(f"✅ Config saved to: {config_path}")

# Step 5: Verify it loads back correctly
print("\nStep 4: Verifying...")
loaded = joblib.load(save_path)
print(f"✅ Loads correctly!")
print(f"✅ Type: {type(loaded)}")

# Step 6: Show file sizes
size = os.path.getsize(save_path) / 1024
print(f"✅ File size: {size:.2f} KB")

print("\n" + "="*60)
print("✅ JOBLIB SERIALIZATION DONE!")
print("="*60)
print("\nFiles created:")
print(f"  ✅ models/feature_engineer.pkl")
print(f"  ✅ models/feature_config.pkl")
print("\nTo use later:")
print("  import joblib")
print("  engineer = joblib.load('models/feature_engineer.pkl')")
print("  features = engineer.engineer_all_features(new_data)")
print("\n🎉 Week 1 = 100% COMPLETE!")
print("="*60)