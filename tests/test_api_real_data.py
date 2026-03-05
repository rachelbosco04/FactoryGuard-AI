import requests
import pandas as pd
import numpy as np
import time

API_URL = "http://localhost:5000"

# Load one row of actual data
print("Loading real sensor data...")
df = pd.read_csv('data/processed/train_features.csv', nrows=1)

# Remove non-feature columns
exclude_cols = ['machine_id', 'timestamp', 'hour_of_operation', 'failure', 'hours_to_failure']
for col in exclude_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Fix NaN and Inf values
df = df.fillna(0)  # Replace NaN with 0
df = df.replace([np.inf, -np.inf], 0)  # Replace Inf with 0

# Convert to dictionary
sensor_data = df.iloc[0].to_dict()

# Convert numpy types to Python native types for JSON compatibility
sensor_data = {k: float(v) for k, v in sensor_data.items()}

print(f"✅ Loaded {len(sensor_data)} features from real data")

# Create payload
payload = {"sensors": sensor_data}

# Test single prediction
print("\n" + "="*70)
print("TESTING WITH REAL DATA")
print("="*70)

start = time.time()
response = requests.post(f"{API_URL}/predict", json=payload)
end = time.time()

response_time_ms = (end - start) * 1000

if response.status_code == 200:
    result = response.json()
    print(f"\n✅ Status: 200 OK")
    print(f"✅ Failure Probability: {result['failure_probability']}")
    print(f"✅ Prediction: {result['prediction']}")
    print(f"⏱️  Response Time: {response_time_ms:.2f} ms")
    
    if response_time_ms < 50:
        print(f"✅ PASS: Response time < 50ms")
    else:
        print(f"❌ FAIL: Response time > 50ms")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.json())

# Test 10 requests for average
print("\n" + "="*70)
print("TESTING 10 REQUESTS")
print("="*70)

times = []
for i in range(10):
    start = time.time()
    response = requests.post(f"{API_URL}/predict", json=payload)
    end = time.time()
    
    if response.status_code == 200:
        times.append((end - start) * 1000)

if times:
    import statistics
    print(f"\n📊 Results:")
    print(f"   Mean: {statistics.mean(times):.2f} ms")
    print(f"   Median: {statistics.median(times):.2f} ms")
    print(f"   Min: {min(times):.2f} ms")
    print(f"   Max: {max(times):.2f} ms")
    
    if statistics.mean(times) < 50:
        print(f"\n✅ PASS: Average < 50ms")
    else:
        print(f"\n❌ FAIL: Average > 50ms")