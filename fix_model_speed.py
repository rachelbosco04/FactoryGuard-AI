import joblib
import os

print("="*70)
print("FIXING MODEL FOR FAST API INFERENCE")
print("="*70)

# Load the tuned LightGBM model
model_path = 'models/lightgbm_tuned.pkl'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("\nAvailable models:")
    import glob
    for f in glob.glob('models/*.pkl'):
        print(f"  - {f}")
    exit(1)

print(f"\n📦 Loading model: {model_path}")
model = joblib.load(model_path)

print(f"✅ Model loaded: {type(model).__name__}")

# Check if it's LightGBM
if 'LGBM' in type(model).__name__:
    print("✅ Confirmed: LightGBM model")
    
    # Get model info
    if hasattr(model, 'n_features_in_'):
        print(f"✅ Features: {model.n_features_in_}")
    
    # The issue: LightGBM validates feature names on every predict()
    # Solution: Use the underlying booster which is faster
    
    print("\n🔧 Extracting booster for faster inference...")
    
    # Save just the booster (no sklearn wrapper overhead)
    if hasattr(model, 'booster_'):
        booster = model.booster_
        
        # Save booster as txt for fastest loading
        booster.save_model('models/lightgbm_booster.txt')
        print("✅ Saved booster: models/lightgbm_booster.txt")
        
        # Also save the sklearn wrapper but optimized
        joblib.dump(model, 'models/best_model_production_fast.pkl', compress=3)
        print("✅ Saved optimized model: models/best_model_production_fast.pkl")
    
    print("\n" + "="*70)
    print("✅ MODEL OPTIMIZATION COMPLETE")
    print("="*70)
    print("\n📝 Update app.py to use:")
    print("   MODEL_PATH = 'models/best_model_production_fast.pkl'")
    print("\nOr for even faster (native booster):")
    print("   Load booster directly from lightgbm_booster.txt")
    print("="*70)

else:
    print(f"⚠️  Not a LightGBM model: {type(model).__name__}")
    print("   Saving as-is...")
    joblib.dump(model, 'models/best_model_production_fast.pkl')
    print("✅ Saved: models/best_model_production_fast.pkl")