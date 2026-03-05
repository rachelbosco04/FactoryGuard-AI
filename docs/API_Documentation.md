# API Documentation

The predictive maintenance model is deployed through a Flask REST API that receives CNC machine sensor data and returns machine failure predictions in real time.

## Base URL

http://localhost:5000

## Health Endpoint

GET /health

Checks whether the API and model are running correctly.

## Prediction Endpoint

POST /predict

Returns machine failure probability based on sensor data.

## Example Request

```json
{
  "sensors": {
    "spindle_temp": 45.2,
    "spindle_vibration": 0.2,
    "spindle_speed": 3000,
    "motor_current": 12.5,
    "tool_vibration": 0.1,
    "tool_wear": 0.15,
    "cutting_force": 150,
    "hydraulic_pressure": 50,
    "coolant_flow": 5,
    "coolant_temp": 25,
    "acoustic_emission": 0.05,
    "power_consumption": 2.5,
    "feed_rate": 500,
    "ambient_temp": 22,
    "ambient_humidity": 45
  }
}