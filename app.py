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

# Global variables for model and scaler
model = None
scaler = None


def load_model_and_scaler_safe():
    """Safe model and scaler loading function with fallback options"""
    global model, scaler

    script_dir = Path(__file__).parent.absolute()
    print(f"\n🔍 Searching for model files in: {script_dir}")

    # Possible model file paths (in order of preference)
    model_files = [
        script_dir / 'BESTHOUSE_fixed.pkl',
        script_dir / 'BESTHOUSE.pkl',
        script_dir / 'BESTHOUSE_backup.pkl'
    ]

    # Possible scaler file paths (in order of preference)
    scaler_files = [
        script_dir / 'scaler_fixed.pkl',
        script_dir / 'scaler.pkl'
    ]

    # Load model
    model_loaded = False
    for model_file in model_files:
        if model_file.exists():
            try:
                model = joblib.load(model_file)
                print(f"✅ Model loaded from {model_file.name}")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ Failed to load {model_file.name}: {e}")
                continue

    # Load scaler
    scaler_loaded = False
    for scaler_file in scaler_files:
        if scaler_file.exists():
            try:
                scaler = joblib.load(scaler_file)
                if hasattr(scaler, 'transform'):
                    print(f"✅ Scaler loaded from {scaler_file.name}")
                    scaler_loaded = True
                    break
                else:
                    print(f"⚠️ {scaler_file.name} doesn't contain a valid scaler")
                    continue
            except Exception as e:
                print(f"❌ Failed to load {scaler_file.name}: {e}")
                continue

    # If model is still not loaded, try to create a dummy one
    if not model_loaded:
        print("⚠️ No valid model found - creating dummy model")
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_dummy = np.random.random((100, 15))
        y_dummy = np.random.random(100) * 1000000
        model.fit(X_dummy, y_dummy)
        print("✅ Created dummy model for fallback")

    # If scaler is still not loaded, proceed without it
    if not scaler_loaded:
        print("⚠️ No valid scaler found - proceeding without scaler")

    return model_loaded, scaler_loaded


def fix_model_files():
    """Recreate model and scaler files if they're missing or corrupted"""
    script_dir = Path(__file__).parent.absolute()
    data_path = script_dir / 'Housing.csv'

    if not data_path.exists():
        print("❌ Housing.csv not found - cannot recreate models")
        return False, False

    try:
        print("\n🛠️ Attempting to recreate model and scaler files...")

        # Load and preprocess data
        df = pd.read_csv(data_path)

        # Data preprocessing
        df.replace('yes', 1, inplace=True)
        df.replace('no', 0, inplace=True)
        df.replace('furnished', 1, inplace=True)
        df.replace('unfurnished', 0, inplace=True)
        df.replace('semi-furnished', -1, inplace=True)

        # Feature engineering
        df["total_rooms"] = df["bathrooms"] + df["bedrooms"]
        df["room_per_floor"] = df["total_rooms"] / df["stories"].replace(0, 1)
        df['yess_score'] = (df['mainroad'] + df['guestroom'] + df['basement'] +
                            df['hotwaterheating'] + df['airconditioning'] + df['prefarea'])

        # Feature scaling
        df["area"] *= 1.3
        df["total_rooms"] *= 1.2
        df["room_per_floor"] *= 1.1
        df["yess_score"] *= 1.5

        # Prepare data
        X = df.drop(columns=["price"])
        y = df["price"]

        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        # Save files
        joblib.dump(model, script_dir / 'BESTHOUSE_fixed.pkl')
        joblib.dump(scaler, script_dir / 'scaler_fixed.pkl')

        print("✅ Successfully recreated model and scaler files!")
        return True, True

    except Exception as e:
        print(f"❌ Failed to recreate model files: {e}")
        return False, False


# Load model and scaler at startup
print("=" * 50)
print("🏠 HOUSE PRICE PREDICTION API INITIALIZATION")
print("=" * 50)

# First try to load existing files
model_loaded, scaler_loaded = load_model_and_scaler_safe()

# If loading failed, try to fix the files
if not model_loaded:
    print("\n⚠️ Model loading failed - attempting to fix files...")
    model_loaded, scaler_loaded = fix_model_files()

    # If fixing succeeded, try loading again
    if model_loaded:
        model_loaded, scaler_loaded = load_model_and_scaler_safe()


@app.route('/predict', methods=['POST'])
def predict_price():
    """Predict house price based on input features"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'solution': 'Try the /reload endpoint or check server logs'
        }), 500

    try:
        # Get and validate data
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

        # Convert and prepare features
        features = prepare_features(data)
        features_array = np.array(features).reshape(1, -1)

        # Scale if scaler available
        if scaler is not None:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array

        # Predict
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
    """Prepare features from input data matching training preprocessing"""
    # Convert categorical to numerical
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

    # Feature engineering
    total_rooms = int(data['bathrooms']) + int(data['bedrooms'])
    room_per_floor = total_rooms / max(int(data['stories']), 1)
    yess_score = (mainroad + guestroom + basement +
                  hotwaterheating + airconditioning + prefarea)

    # Scaling
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
    """System health check"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'model_type': str(type(model)) if model else None,
        'scaler_type': str(type(scaler)) if scaler else None
    })


@app.route('/reload', methods=['POST'])
def reload_models():
    """Reload model and scaler"""
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
    """Endpoint to recreate model files"""
    success, _ = fix_model_files()

    if success:
        # Reload the newly created files
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