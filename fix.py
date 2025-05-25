import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os


def diagnose_and_fix_models():
    """Diagnose and fix the corrupted model files"""
    print("🔧 DIAGNOSING AND FIXING MODEL FILES")
    print("=" * 60)

    # Check if files exist
    model_exists = os.path.exists('BESTHOUSE.pkl')
    scaler_exists = os.path.exists('scaler.pkl')
    data_exists = os.path.exists('Housing.csv')

    print(f"Files status:")
    print(f"  BESTHOUSE.pkl: {'✅ Exists' if model_exists else '❌ Missing'}")
    print(f"  scaler.pkl: {'✅ Exists' if scaler_exists else '❌ Missing'}")
    print(f"  Housing.csv: {'✅ Exists' if data_exists else '❌ Missing'}")

    # Try to load and diagnose current files
    model = None
    scaler = None

    if model_exists:
        try:
            with open('BESTHOUSE.pkl', 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully with pickle")
        except Exception as e:
            print(f"❌ Model loading with pickle failed: {e}")
            try:
                model = joblib.load('BESTHOUSE.pkl')
                print("✅ Model loaded successfully with joblib")
            except Exception as e2:
                print(f"❌ Model loading with joblib also failed: {e2}")

    if scaler_exists:
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler loaded: type = {type(scaler)}")
            if isinstance(scaler, np.ndarray):
                print("⚠️  Scaler is a numpy array instead of StandardScaler object")
        except Exception as e:
            print(f"❌ Scaler loading failed: {e}")

    # If model is corrupted or missing, recreate it
    if model is None and data_exists:
        print("\n🔄 RECREATING MODEL FROM SCRATCH")
        print("-" * 40)

        # Load and preprocess data
        df = pd.read_csv('Housing.csv')
        print(f"Data loaded: {df.shape}")

        # Data preprocessing (from your original code)
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

        # Feature scaling from your original code
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

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train multiple models to find the best one
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(tol=1e-3),
            'Lasso': Lasso(tol=1e-3),
            'ElasticNet': ElasticNet(tol=1e-3),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100)
        }

        best_model = None
        best_score = -float('inf')
        best_name = ""

        print("\nTraining models...")
        for name, model_obj in models.items():
            try:
                model_obj.fit(X_train, y_train)
                y_pred = model_obj.predict(X_test)
                score = r2_score(y_test, y_pred)
                print(f"  {name}: R² = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = model_obj
                    best_name = name
            except Exception as e:
                print(f"  {name}: Failed - {e}")

        print(f"\n🏆 Best model: {best_name} (R² = {best_score:.4f})")
        model = best_model

    # Save the working model and scaler
    if model is not None:
        try:
            # Save with joblib (recommended for sklearn models)
            joblib.dump(model, 'BESTHOUSE_fixed.pkl')
            print("✅ Model saved as BESTHOUSE_fixed.pkl")

            # Also save with pickle as backup
            with open('BESTHOUSE_backup.pkl', 'wb') as f:
                pickle.dump(model, f)
            print("✅ Model backup saved as BESTHOUSE_backup.pkl")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")

    if scaler is not None and hasattr(scaler, 'transform'):
        try:
            joblib.dump(scaler, 'scaler_fixed.pkl')
            print("✅ Scaler saved as scaler_fixed.pkl")
        except Exception as e:
            print(f"❌ Failed to save scaler: {e}")
    elif data_exists and model is not None:
        # Recreate scaler if it was corrupted
        print("🔄 Recreating scaler...")
        df = pd.read_csv('Housing.csv')
        # Apply same preprocessing
        df.replace('yes', 1, inplace=True)
        df.replace('no', 0, inplace=True)
        df.replace('furnished', 1, inplace=True)
        df.replace('unfurnished', 0, inplace=True)
        df.replace('semi-furnished', -1, inplace=True)

        df["total_rooms"] = df["bathrooms"] + df["bedrooms"]
        df["room_per_floor"] = df["total_rooms"] / df["stories"].replace(0, 1)
        df['yess_score'] = (df['mainroad'] + df['guestroom'] + df['basement'] +
                            df['hotwaterheating'] + df['airconditioning'] + df['prefarea'])

        df["area"] *= 1.3
        df["total_rooms"] *= 1.2
        df["room_per_floor"] *= 1.1
        df["yess_score"] *= 1.5

        X = df.drop(columns=["price"])
        new_scaler = StandardScaler()
        new_scaler.fit(X)

        joblib.dump(new_scaler, 'scaler_fixed.pkl')
        print("✅ New scaler created and saved as scaler_fixed.pkl")

    return model, scaler


def test_fixed_models():
    """Test the fixed model files"""
    print("\n🧪 TESTING FIXED MODEL FILES")
    print("=" * 40)

    try:
        model = joblib.load('BESTHOUSE_fixed.pkl')
        print(f"✅ Fixed model loaded: {type(model)}")
    except Exception as e:
        print(f"❌ Failed to load fixed model: {e}")
        return False, False

    try:
        scaler = joblib.load('scaler_fixed.pkl')
        print(f"✅ Fixed scaler loaded: {type(scaler)}")

        # Test scaler functionality
        if hasattr(scaler, 'transform'):
            print("✅ Scaler has transform method")
            # Test with dummy data
            test_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
            try:
                scaled_data = scaler.transform(test_data)
                print("✅ Scaler transform works")
            except Exception as e:
                print(f"❌ Scaler transform failed: {e}")
        else:
            print("❌ Scaler missing transform method")
            return True, False

    except Exception as e:
        print(f"❌ Failed to load fixed scaler: {e}")
        return True, False

    return True, True


def create_safe_loader():
    """Create a safe model loading function"""
    code = '''
def load_model_and_scaler_safe():
    """Safe model and scaler loading function"""
    import joblib
    import os

    model = None
    scaler = None

    # Try to load fixed versions first
    model_files = ['BESTHOUSE_fixed.pkl', 'BESTHOUSE.pkl', 'BESTHOUSE_backup.pkl']
    scaler_files = ['scaler_fixed.pkl', 'scaler.pkl']

    # Load model
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                print(f"✅ Model loaded from {model_file}")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
                continue

    # Load scaler
    for scaler_file in scaler_files:
        if os.path.exists(scaler_file):
            try:
                scaler = joblib.load(scaler_file)
                if hasattr(scaler, 'transform'):
                    print(f"✅ Scaler loaded from {scaler_file}")
                    break
                else:
                    print(f"⚠️  {scaler_file} doesn't contain a valid scaler")
                    continue
            except Exception as e:
                print(f"❌ Failed to load {scaler_file}: {e}")
                continue

    return model, scaler

# Test the safe loader
if __name__ == "__main__":
    model, scaler = load_model_and_scaler_safe()
    print(f"Model: {type(model) if model else None}")
    print(f"Scaler: {type(scaler) if scaler else None}")
'''

    with open('safe_loader.py', 'w') as f:
        f.write(code)
    print("✅ Safe loader created as 'safe_loader.py'")


if __name__ == "__main__":
    # Run the diagnostic and fix
    model, scaler = diagnose_and_fix_models()

    # Test the fixed files
    model_ok, scaler_ok = test_fixed_models()

    # Create safe loader
    create_safe_loader()

    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    if model_ok and scaler_ok:
        print("✅ All files fixed successfully!")
        print("📁 Use these files in your app:")
        print("   - BESTHOUSE_fixed.pkl (model)")
        print("   - scaler_fixed.pkl (scaler)")
        print("📄 Use safe_loader.py for reliable loading")
    else:
        print("❌ Some issues remain. Check the output above.")
        print("💡 Make sure Housing.csv is available to recreate models.")