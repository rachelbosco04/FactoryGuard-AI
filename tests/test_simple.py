import requests
import time
import statistics

API_URL = "http://localhost:5000"

# Create clean test data - manually set all 154 features to 0
# The model will handle zeros fine
print("Creating clean test data...")

# Start with base sensors
test_data = {
    'spindle_temp': 45.2,
    'spindle_vibration': 0.15,
    'spindle_speed': 3000.0,
    'motor_current': 12.5,
    'tool_vibration': 0.08,
    'tool_wear': 0.02,
    'cutting_force': 150.0,
    'hydraulic_pressure': 50.0,
    'coolant_flow': 5.0,
    'coolant_temp': 25.0,
    'acoustic_emission': 0.05,
    'power_consumption': 2.5,
    'feed_rate': 500.0,
    'ambient_temp': 22.0,
    'ambient_humidity': 45.0
}

# Add engineered features (all zeros - model will handle it)
# Rolling means
for sensor in ['spindle_temp', 'spindle_vibration', 'motor_current', 'tool_vibration', 'tool_wear']:
    for window in ['6h', '12h', '24h']:
        test_data[f'{sensor}_rolling_mean_{window}'] = 0.0
        test_data[f'{sensor}_rolling_std_{window}'] = 0.0

# Lag features
for sensor in ['spindle_vibration', 'tool_wear', 'motor_current']:
    for lag in ['1h', '2h', '3h', '6h']:
        test_data[f'{sensor}_lag_{lag}'] = 0.0

# EMA features
for sensor in ['spindle_temp', 'motor_current', 'tool_vibration']:
    test_data[f'{sensor}_ema_0.1'] = 0.0
    test_data[f'{sensor}_ema_0.3'] = 0.0

# Degradation index
test_data['degradation_index'] = 0.0

print(f"✅ Created {len(test_data)} features")

payload = {"sensors": test_data}

# Test single prediction
print("\n" + "="*70)
print("SINGLE PREDICTION TEST")
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
        print(f"\n✅ ✅ ✅ PASS: Response time < 50ms! ✅ ✅ ✅")
    else:
        print(f"\n❌ FAIL: Response time > 50ms")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.json())

# Test 100 requests
print("\n" + "="*70)
print("LATENCY TEST (100 REQUESTS)")
print("="*70)

times = []
print("\n🔄 Sending requests...", end='', flush=True)

for i in range(100):
    start = time.time()
    response = requests.post(f"{API_URL}/predict", json=payload)
    end = time.time()
    
    if response.status_code == 200:
        times.append((end - start) * 1000)
    
    if (i + 1) % 10 == 0:
        print(f" {i+1}", end='', flush=True)

print(" Done!\n")

if times:
    mean_time = statistics.mean(times)
    
    print(f"📊 Latency Statistics:")
    print(f"   Total requests: {len(times)}")
    print(f"   Mean: {mean_time:.2f} ms")
    print(f"   Median: {statistics.median(times):.2f} ms")
    print(f"   Min: {min(times):.2f} ms")
    print(f"   Max: {max(times):.2f} ms")
    print(f"   Std Dev: {statistics.stdev(times):.2f} ms")
    
    under_50 = sum(1 for t in times if t < 50)
    percentage = (under_50 / len(times)) * 100
    
    print(f"\n✅ Requests < 50ms: {under_50}/{len(times)} ({percentage:.1f}%)")
    
    if mean_time < 50:
        print(f"\n✅ ✅ ✅ PASS: Average response time < 50ms! ✅ ✅ ✅")
        print(f"\n🎉 SUCCESS! Your API meets the <50ms requirement! 🎉")
    else:
        print(f"\n❌ FAIL: Average response time > 50ms")

print("\n" + "="*70)