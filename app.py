import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model artifacts
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "ensemble_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

model = None
scaler = None
feature_names = None

def load_models():
    global model, scaler, feature_names
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feature_names = joblib.load(FEATURES_PATH)
            return True
    except Exception as e:
        print(f"Error loading models: {e}")
    return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        if not load_models():
            return jsonify({'error': 'Models not loaded yet. Please wait or train the model.'}), 500

    try:
        data = request.json
        # Map input to feature names
        input_data = {}
        
        # Base features
        input_data['Rainfall_mm'] = float(data.get('Rainfall_mm', 0))
        input_data['Slope_Angle'] = float(data.get('Slope_Angle', 0))
        input_data['Soil_Saturation'] = float(data.get('Soil_Saturation', 0))
        input_data['Vegetation_Cover'] = float(data.get('Vegetation_Cover', 0))
        input_data['Earthquake_Activity'] = float(data.get('Earthquake_Activity', 0))
        input_data['Proximity_to_Water'] = float(data.get('Proximity_to_Water', 0))
        
        # Enhanced features (calculated in the backend as per train_model.py)
        elevation = float(data.get('Elevation', (input_data['Slope_Angle'] * 25 + 500)))
        input_data['Elevation'] = elevation
        input_data['Curvature'] = float(data.get('Curvature', input_data['Slope_Angle'] * 0.01))
        input_data['Drainage_Density'] = float(data.get('Drainage_Density', 4.0))
        input_data['Distance_to_River'] = float(data.get('Distance_to_River', (1 - input_data['Proximity_to_Water']) * 5))
        input_data['Rainfall_24h'] = input_data['Rainfall_mm'] * 0.4
        input_data['Rainfall_48h'] = input_data['Rainfall_mm'] * 0.7
        input_data['Rainfall_72h'] = input_data['Rainfall_mm'] * 0.9
        
        # Handle LULC
        lulc = data.get('LULC', 'Forest')
        for cat in ['Forest', 'Agriculture', 'Barren', 'Urban', 'Grassland']:
            input_data[f'LULC_{cat}'] = 1.0 if lulc == cat else 0.0
            
        # Engineered features
        input_data['Slope_Elevation_Interaction'] = input_data['Slope_Angle'] * input_data['Elevation'] / 1000
        input_data['Slope_Curvature_Interaction'] = input_data['Slope_Angle'] * input_data['Curvature']
        input_data['Rainfall_Saturation'] = input_data['Rainfall_mm'] * input_data['Soil_Saturation']
        input_data['Water_Proximity_Drainage'] = input_data['Proximity_to_Water'] * input_data['Drainage_Density']
        input_data['Rainfall_Intensity_24h'] = input_data['Rainfall_24h'] / 24
        input_data['Rainfall_Intensity_Change'] = (input_data['Rainfall_72h'] - input_data['Rainfall_24h']) / 48
        input_data['Veg_Slope_Risk'] = (1 - input_data['Vegetation_Cover']) * input_data['Slope_Angle']
        input_data['Risk_Composite'] = (
            input_data['Rainfall_mm'] / 300 * 0.3 +
            input_data['Slope_Angle'] / 60 * 0.25 +
            input_data['Soil_Saturation'] * 0.2 +
            (1 - input_data['Vegetation_Cover']) * 0.15 +
            input_data['Earthquake_Activity'] / 7 * 0.1
        )

        # Create feature vector in correct order
        features = []
        for name in feature_names:
            features.append(input_data.get(name, 0.0))
            
        # Scale and predict
        features_scaled = scaler.transform([features])
        proba = model.predict_proba(features_scaled)[0][1]
        
        # Categorize risk
        if proba < 0.2:
            risk = "Very Low"
        elif proba < 0.4:
            risk = "Low"
        elif proba < 0.6:
            risk = "Moderate"
        elif proba < 0.8:
            risk = "High"
        else:
            risk = "Very High"
            
        return jsonify({
            'probability': proba,
            'risk_level': risk,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000)
