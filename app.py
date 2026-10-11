# Exercise 2: Flask API. Load the model once at startup.
# POST /predict -> study_hours + attendance_percentage (validate, 400 on errors)
import json
from flask import Flask, request, jsonify
import joblib
import pandas as pd

FEATURES = ['study_hours', 'attendance_percentage']
MODEL_PATH = 'student_pass_model.joblib'

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def _error(msg):
    return jsonify({'error': msg}), 400


@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return _error('Request body must be JSON')
    try:
        data = request.get_json(force=False)
    except Exception:
        return _error('Invalid JSON')
    if not isinstance(data, dict):
        return _error('Request body must be a JSON object')
    if 'study_hours' not in data:
        return _error('study_hours is required')
    if 'attendance_percentage' not in data:
        return _error('attendance_percentage is required')
    sh = data['study_hours']
    att = data['attendance_percentage']
    if isinstance(sh, bool) or isinstance(att, bool):
        return _error('study_hours and attendance_percentage must be numeric')
    if isinstance(sh, str) or isinstance(att, str):
        return _error('study_hours and attendance_percentage must be numeric')
    try:
        shv = float(sh)
        atv = float(att)
    except Exception:
        return _error('study_hours and attendance_percentage must be numeric')
    if shv < 0 or shv > 24:
        return _error('study_hours must be between 0 and 24')
    if atv < 0 or atv > 100:
        return _error('attendance_percentage must be between 0 and 100')
    X = pd.DataFrame([[shv, atv]], columns=FEATURES)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    return jsonify({
        'study_hours': shv,
        'attendance_percentage': atv,
        'prediction': pred,
        'result': 'Pass' if pred == 1 else 'Fail',
        'pass_probability': prob
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)
# Test client results (5 cases)
# 1. valid (5,85) -> 200 {"study_hours":5.0,"attendance_percentage":85.0,"prediction":1,"result":"Pass",...}
# 2. attendance 120 -> 400 {"error":"attendance_percentage must be between 0 and 100"}
# 3. attendance -5 -> 400 {"error":"attendance_percentage must be between 0 and 100"}
# 4. missing attendance -> 400 {"error":"attendance_percentage is required"}
# 5. study_hours "abc" -> 400 {"error":"study_hours and attendance_percentage must be numeric"}
# curl.exe form (PowerShell-safe, use port 8000)
# '{"study_hours":5,"attendance_percentage":85}' | Set-Content body.json -Encoding ascii
# curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@body.json"
# (replace body.json contents for other cases)
# Test client results (5 cases)
# 1: 200 {'attendance_percentage': 85.0, 'pass_probability': 0.9899160914187777, 'prediction': 1, 'result': 'Pass', 'study_hours': 5.0}
# 2: 400 {'error': 'attendance_percentage must be between 0 and 100'}
# 3: 400 {'error': 'attendance_percentage must be between 0 and 100'}
# 4: 400 {'error': 'attendance_percentage is required'}
# 5: 400 {'error': 'study_hours and attendance_percentage must be numeric'}
# curl.exe form (PowerShell-safe, use port 8000)
# '{"study_hours":5,"attendance_percentage":85}' | Set-Content body.json -Encoding ascii
# curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@body.json"
# Test client results (5 cases)
# 1: 200 {'attendance_percentage': 85.0, 'pass_probability': 0.8105016824526622, 'prediction': 1, 'result': 'Pass', 'study_hours': 5.0}
# 2: 400 {'error': 'attendance_percentage must be between 0 and 100'}
# 3: 400 {'error': 'attendance_percentage must be between 0 and 100'}
# 4: 400 {'error': 'attendance_percentage is required'}
# 5: 400 {'error': 'study_hours and attendance_percentage must be numeric'}
# curl.exe form (PowerShell-safe, use port 8000)
# '{"study_hours":5,"attendance_percentage":85}' | Set-Content body.json -Encoding ascii
# curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@body.json"

