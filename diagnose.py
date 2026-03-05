import time
import joblib
import numpy as np

print("="*70)
print("DIAGNOSTIC TEST - FINDING THE BOTTLENECK")
print("="*70)

# Test 1: Load model
print("\n1. Loading model...")
start = time.time()
model = joblib.load('models/best_model_production.pkl')
load_time = (time.time() - start) * 1000
print(f"   Load time: {load_time:.2f} ms")
print(f"   Model type: {type(model).__name__}")

# Test 2: Create dummy data
print("\n2. Creating test data...")
start = time.time()
X_test = np.zeros((1, 154), dtype=np.float32)
data_time = (time.time() - start) * 1000
print(f"   Data creation time: {data_time:.2f} ms")

# Test 3: First prediction (cold start)
print("\n3. First prediction (cold start)...")
start = time.time()
pred = model.predict_proba(X_test)
first_pred_time = (time.time() - start) * 1000
print(f"   First prediction time: {first_pred_time:.2f} ms")
print(f"   Result: {pred}")

# Test 4: Second prediction (warm)
print("\n4. Second prediction (warm)...")
start = time.time()
pred = model.predict_proba(X_test)
second_pred_time = (time.time() - start) * 1000
print(f"   Second prediction time: {second_pred_time:.2f} ms")

# Test 5: 10 predictions
print("\n5. Testing 10 predictions...")
times = []
for i in range(10):
    start = time.time()
    pred = model.predict_proba(X_test)
    times.append((time.time() - start) * 1000)

import statistics
print(f"   Mean: {statistics.mean(times):.2f} ms")
print(f"   Min: {min(times):.2f} ms")
print(f"   Max: {max(times):.2f} ms")

# Test 6: Check if it's the booster
print("\n6. Testing direct booster (if available)...")
if hasattr(model, 'booster_'):
    start = time.time()
    pred = model.booster_.predict(X_test)
    booster_time = (time.time() - start) * 1000
    print(f"   Booster prediction time: {booster_time:.2f} ms")
else:
    print("   No booster available")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
print(f"Model load: {load_time:.2f} ms")
print(f"First prediction: {first_pred_time:.2f} ms")
print(f"Warm predictions: {statistics.mean(times):.2f} ms average")

if statistics.mean(times) > 1000:
    print("\n❌ PROBLEM IDENTIFIED: Model predictions are inherently slow")
    print("   This suggests:")
    print("   1. Wrong model file (maybe Random Forest, not LightGBM?)")
    print("   2. Model is too complex")
    print("   3. Hardware limitation")
    print("\n   Check: What model is actually in best_model_production.pkl?")
else:
    print("\n✅ Model predictions are fast!")
    print("   Problem is elsewhere (likely Flask/network overhead)")