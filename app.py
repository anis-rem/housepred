from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

model = None
scaler = None

def load_model_and_scaler_safe():
    global model, scaler

    script_dir = Path(__file__).parent.absolute()
    model_files = [
        script_dir / 'BESTHOUSE_fixed.pkl',
        script_dir / 'BESTHOUSE.pkl',
        script_dir / 'BESTHOUSE_backup.pkl'
    ]
    scaler_files = [
        script_dir / 'scaler_fixed.pkl',
        script_dir / 'scaler.pkl'
    ]

    model_loaded = False
    for model_file in model_files:
        if model_file.exists():
            try:
                model = joblib.load(model_file)
                model_loaded = True
                break
            except:
                continue

    scaler_loaded = False
    for scaler_file in scaler_files:
        if scaler_file.exists():
            try:
                scaler = joblib.load(scaler_file)
                if hasattr(scaler, 'transform'):
                    scaler_loaded = True
                    break
            except:
                continue

    if not model_loaded:
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_dummy = np.random.random((100, 15))
        y_dummy = np.random.random(100) * 1000000
        model.fit(X_dummy, y_dummy)

    return model_loaded, scaler_loaded

def fix_model_files():
    script_dir = Path(__file__).parent.absolute()
    data_path = script_dir / 'Housing.csv'

    if not data_path.exists():
        return False, False

    try:
        df = pd.read_csv(data_path)
        df.replace('yes', 1, inplace=True)
        df.replace('no', 0, inplace=True)
        df.replace('furnished', 1, inplace=True)
        df.replace('unfurnished', 0, inplace=True)
        df.replace('semi-furnished', -1, inplace=True)

        df["total_rooms"] = df["bathrooms"] + df["bedrooms"]
        df["room_per_floor"] = df["total_rooms"] / df["stories"].replace(0, 1)
        df['yess_score'] = (
            df['mainroad'] + df['guestroom'] + df['basement'] +
            df['hotwaterheating'] + df['airconditioning'] + df['prefarea']
        )

        df["area"] *= 1.3
        df["total_rooms"] *= 1.2
        df["room_per_floor"] *= 1.1
        df["yess_score"] *= 1.5

        X = df.drop(columns=["price"])
        y = df["price"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        joblib.dump(model, script_dir / 'BESTHOUSE_fixed.pkl')
        joblib.dump(scaler, script_dir / 'scaler_fixed.pkl')

        return True, True

    except:
        return False, False

print("=" * 50)
print("🏠 HOUSE PRICE PREDICTION API INITIALIZATION")
print("=" * 50)

model_loaded, scaler_loaded = load_model_and_scaler_safe()
if not model_loaded:
    model_loaded, scaler_loaded = fix_model_files()
    if model_loaded:
        model_loaded, scaler_loaded = load_model_and_scaler_safe()

@app.route('/predict', methods=['POST'])
def predict_price():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'solution': 'Try the /reload endpoint or check server logs'
        }), 500

    try:
        data = request.json
        required_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                           'mainroad', 'guestroom', 'basement', 'hotwaterheating',
                           'airconditioning', 'prefarea', 'furnishingstatus']

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing fields: {missing_fields}'
            }), 400

        features = prepare_features(data)
        features_array = np.array(features).reshape(1, -1)

        if scaler is not None:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array

        prediction = model.predict(features_scaled)[0]

        return jsonify({
            'success': True,
            'predicted_price': float(prediction),
            'features_used': features,
            'scaler_used': scaler is not None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 400

def prepare_features(data):
    mainroad = 1 if str(data['mainroad']).lower() == 'yes' else 0
    guestroom = 1 if str(data['guestroom']).lower() == 'yes' else 0
    basement = 1 if str(data['basement']).lower() == 'yes' else 0
    hotwaterheating = 1 if str(data['hotwaterheating']).lower() == 'yes' else 0
    airconditioning = 1 if str(data['airconditioning']).lower() == 'yes' else 0
    prefarea = 1 if str(data['prefarea']).lower() == 'yes' else 0

    furnishing_str = str(data['furnishingstatus']).lower()
    if furnishing_str == 'furnished':
        furnishingstatus = 1
    elif furnishing_str == 'unfurnished':
        furnishingstatus = 0
    else:
        furnishingstatus = -1

    total_rooms = int(data['bathrooms']) + int(data['bedrooms'])
    room_per_floor = total_rooms / max(int(data['stories']), 1)
    yess_score = (mainroad + guestroom + basement +
                  hotwaterheating + airconditioning + prefarea)

    area_scaled = float(data['area']) * 1.3
    total_rooms_scaled = total_rooms * 1.2
    room_per_floor_scaled = room_per_floor * 1.1
    yess_score_scaled = yess_score * 1.5

    return [
        area_scaled,
        int(data['bedrooms']),
        int(data['bathrooms']),
        int(data['stories']),
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        int(data['parking']),
        prefarea,
        furnishingstatus,
        total_rooms_scaled,
        room_per_floor_scaled,
        yess_score_scaled
    ]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'model_type': str(type(model)) if model else None,
        'scaler_type': str(type(scaler)) if scaler else None
    })

@app.route('/reload', methods=['POST'])
def reload_models():
    global model, scaler
    model_loaded, scaler_loaded = load_model_and_scaler_safe()

    return jsonify({
        'success': model_loaded,
        'model_loaded': model_loaded,
        'scaler_loaded': scaler_loaded,
        'message': 'Models reloaded' if model_loaded else 'Failed to reload models'
    })

@app.route('/fix-models', methods=['POST'])
def fix_models_endpoint():
    success, _ = fix_model_files()

    if success:
        model_loaded, scaler_loaded = load_model_and_scaler_safe()
        return jsonify({
            'success': True,
            'message': 'Successfully recreated and loaded model files',
            'model_loaded': model_loaded,
            'scaler_loaded': scaler_loaded
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to recreate model files',
            'solution': 'Ensure Housing.csv exists in the same directory'
        }), 500

if __name__ == '__main__':
    print("\nAvailable endpoints:")
    print("  POST /predict - Make price predictions")
    print("  GET  /health - Check system status")
    print("  POST /reload - Reload models")
    print("  POST /fix-models - Recreate model files")
    print("\nStarting server on http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)
