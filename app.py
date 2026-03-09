from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import shap
import time
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
explainer = None
model_loaded = False
error_message = None


def load_model_safe():
    """Safely load model with proper error handling"""
    global model, explainer, model_loaded, error_message

    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    try:
        import numpy as np
        import lightgbm as lgb

        MODEL_PATH = "models/lightgbm_booster.txt"

        if not os.path.exists(MODEL_PATH):
            error_message = f"Model file not found: {MODEL_PATH}"
            print(f"❌ {error_message}")
            return False

        # Load LightGBM Booster model
        model = lgb.Booster(model_file=MODEL_PATH)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        model_loaded = True

        print("✅ LightGBM model loaded successfully")
        print("✅ SHAP explainer initialized")
        print("=" * 70)

        return True

    except Exception as e:
        error_message = f"Error loading model: {str(e)}"
        print(f"❌ {error_message}")
        print("=" * 70)
        return False


# 🔹 LOAD MODEL WHEN SERVER STARTS (important for Render/Gunicorn)
load_model_safe()


@app.route("/")
def dashboard():
    """Serve Dashboard UI"""
    return send_from_directory(".", "Dashboard.html")


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'error': error_message if not model_loaded else None
    }), 200 if model_loaded else 503


@app.route('/predict', methods=['POST'])
def predict():

    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. ' + (error_message or 'Unknown error'),
            'status': 'error'
        }), 503

    start_time = time.time()

    try:
        import numpy as np

        data = request.get_json()

        if not data or 'sensors' not in data:
            return jsonify({
                'error': 'Missing sensors data',
                'status': 'error'
            }), 400

        sensors = data['sensors']

        # Create feature array
        feature_array = np.zeros(154, dtype=np.float32)

        sensor_mapping = {
            'spindle_temp': 0,
            'spindle_vibration': 1,
            'spindle_speed': 2,
            'motor_current': 3,
            'tool_vibration': 4,
            'tool_wear': 5,
            'cutting_force': 6,
            'hydraulic_pressure': 7,
            'coolant_flow': 8,
            'coolant_temp': 9,
            'acoustic_emission': 10,
            'power_consumption': 11,
            'feed_rate': 12,
            'ambient_temp': 13,
            'ambient_humidity': 14
        }

        for sensor_name, sensor_value in sensors.items():
            if sensor_name in sensor_mapping:
                try:
                    idx = sensor_mapping[sensor_name]
                    feature_array[idx] = float(sensor_value)
                except:
                    pass

        # Feature Engineering
        spindle_temp = feature_array[0]
        spindle_vibration = feature_array[1]
        tool_wear = feature_array[5]

        feature_array[15] = spindle_temp * spindle_vibration
        feature_array[16] = spindle_vibration ** 2
        feature_array[17] = spindle_temp / (tool_wear + 0.001)
        feature_array[18] = spindle_vibration * tool_wear
        feature_array[19] = spindle_temp - 25

        features_2d = feature_array.reshape(1, -1)

        prediction_proba = model.predict(features_2d)
        failure_probability = float(prediction_proba[0])

        # SHAP Explainability
        try:
            shap_values = explainer.shap_values(features_2d)
        except Exception as e:
            print("SHAP error:", e)
            shap_values = [[0]*154]

        shap_importance = list(zip(range(len(shap_values[0])), shap_values[0]))
        shap_importance = sorted(shap_importance, key=lambda x: abs(x[1]), reverse=True)[:5]

        top_features = [
            {"feature_index": int(idx), "impact": float(val)}
            for idx, val in shap_importance
        ]

        response_time_ms = (time.time() - start_time) * 1000

        response = {
            'failure_probability': round(failure_probability, 6),
            'prediction': "failure" if failure_probability > 0.15 else "normal",
            'confidence': round(
                1 - failure_probability if failure_probability < 0.5 else failure_probability, 4
            ),
            'top_feature_impacts': top_features,
            'response_time_ms': round(response_time_ms, 2),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'error': f'Request error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("STARTING FLASK API")
    print("=" * 70)

    port = int(os.environ.get("PORT", 5000))

    print(f"\n🚀 API running at: http://localhost:{port}")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)