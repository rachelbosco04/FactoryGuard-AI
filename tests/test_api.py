"""
API Test Script - Week 4
=========================
Test the Flask API endpoints and measure response time

Author: Your Name
Date: March 2026
"""

import requests
import json
import time
import statistics


API_URL = "http://localhost:5000"


def test_health_check():
    """Test health check endpoint"""
    print("="*70)
    print("TEST 1: HEALTH CHECK")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*70)
    print("TEST 2: MODEL INFO")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/model/info")
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_prediction():
    """Test prediction endpoint"""
    print("\n" + "="*70)
    print("TEST 3: PREDICTION")
    print("="*70)
    
    # Sample sensor data
    payload = {
        "sensors": {
            "spindle_temp": 45.2,
            "spindle_vibration": 0.15,
            "spindle_speed": 3000,
            "motor_current": 12.5,
            "tool_vibration": 0.08,
            "tool_wear": 0.02,
            "cutting_force": 150,
            "hydraulic_pressure": 50,
            "coolant_flow": 5.0,
            "coolant_temp": 25,
            "acoustic_emission": 0.05,
            "power_consumption": 2.5,
            "feed_rate": 500,
            "ambient_temp": 22,
            "ambient_humidity": 45
        }
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            data = response.json()
            response_time = data.get('response_time_ms', 0)
            print(f"\n⏱️  Response Time: {response_time:.2f} ms")
            
            if response_time < 50:
                print(f"✅ PASS: Response time < 50ms ✅")
            else:
                print(f"⚠️  WARNING: Response time > 50ms")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_latency(n_requests=100):
    """Test API latency with multiple requests"""
    print("\n" + "="*70)
    print(f"TEST 4: LATENCY TEST ({n_requests} requests)")
    print("="*70)
    
    payload = {
        "sensors": {
            "spindle_temp": 45.2,
            "spindle_vibration": 0.15,
            "spindle_speed": 3000,
            "motor_current": 12.5,
            "tool_vibration": 0.08,
            "tool_wear": 0.02,
            "cutting_force": 150,
            "hydraulic_pressure": 50,
            "coolant_flow": 5.0,
            "coolant_temp": 25,
            "acoustic_emission": 0.05,
            "power_consumption": 2.5,
            "feed_rate": 500,
            "ambient_temp": 22,
            "ambient_humidity": 45
        }
    }
    
    response_times = []
    
    print(f"\n🔄 Sending {n_requests} requests...")
    
    for i in range(n_requests):
        try:
            start = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            end = time.time()
            
            if response.status_code == 200:
                response_times.append((end - start) * 1000)
            
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{n_requests}")
        
        except Exception as e:
            print(f"   ❌ Request {i+1} failed: {e}")
    
    if response_times:
        print(f"\n📊 Latency Statistics:")
        print(f"   Total requests: {len(response_times)}")
        print(f"   Mean: {statistics.mean(response_times):.2f} ms")
        print(f"   Median: {statistics.median(response_times):.2f} ms")
        print(f"   Min: {min(response_times):.2f} ms")
        print(f"   Max: {max(response_times):.2f} ms")
        print(f"   Std Dev: {statistics.stdev(response_times):.2f} ms")
        
        # Check if under 50ms
        under_50ms = sum(1 for t in response_times if t < 50)
        percentage = (under_50ms / len(response_times)) * 100
        
        print(f"\n✅ Requests < 50ms: {under_50ms}/{len(response_times)} ({percentage:.1f}%)")
        
        if statistics.mean(response_times) < 50:
            print(f"✅ PASS: Average response time < 50ms ✅")
        else:
            print(f"⚠️  FAIL: Average response time > 50ms")
    
    return True


def test_invalid_input():
    """Test API with invalid input"""
    print("\n" + "="*70)
    print("TEST 5: INVALID INPUT HANDLING")
    print("="*70)
    
    # Test 1: Missing sensors field
    print("\n🧪 Test 5.1: Missing 'sensors' field")
    payload = {"data": {}}
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 2: Missing sensor
    print("\n🧪 Test 5.2: Missing sensor")
    payload = {
        "sensors": {
            "spindle_temp": 45.2
            # Missing other sensors
        }
    }
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 3: Invalid value type
    print("\n🧪 Test 5.3: Invalid value type")
    payload = {
        "sensors": {
            "spindle_temp": "invalid",
            "spindle_vibration": 0.15
        }
    }
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    return True


def main():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("API TESTING SUITE - WEEK 4")
    print("="*70)
    print(f"\nTarget API: {API_URL}")
    
    # Check if API is running
    print("\n🔍 Checking if API is running...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        print("✅ API is running!")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: API is not running!")
        print("\n📝 To start the API, run in a separate terminal:")
        print("   python src/api/app.py")
        print("\n⏱️  Wait for 'Running on http://127.0.0.1:5000' message")
        print("   Then run this test script again.\n")
        return
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    input("\nAPI is ready! Press Enter to start tests...")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Prediction", test_prediction),
        ("Latency Test", lambda: test_latency(100)),
        ("Invalid Input", test_invalid_input)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()