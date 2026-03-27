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
        
        # Base features from user input
        rainfall = float(data.get('Rainfall_mm', 1000))
        slope = float(data.get('Slope_Angle', 30))
        elevation = float(data.get('Elevation', 1500))
        veg_cover = float(data.get('Vegetation_Cover', 0.5))
        soil_sat = float(data.get('Soil_Saturation', 0.5))
        curvature = float(data.get('Curvature', 0.0))
        earthquake = float(data.get('Earthquake_Activity', 1.0))
        prox_water = float(data.get('Proximity_to_Water', 0.5))
        
        input_data['Rainfall_mm'] = rainfall
        input_data['Slope_Angle'] = slope
        input_data['Elevation'] = elevation
        input_data['Vegetation_Cover'] = veg_cover
        input_data['Soil_Saturation'] = soil_sat
        input_data['Curvature'] = curvature
        input_data['Earthquake_Activity'] = earthquake
        input_data['Proximity_to_Water'] = prox_water
        
        # New features from GEE dataset (defaults if not in JSON)
        input_data['TRI'] = float(data.get('TRI', 25.0))
        input_data['TWI'] = float(data.get('TWI', 10.0))
        input_data['Aspect'] = float(data.get('Aspect', 180.0))
        
        # Engineered features (must match train_model.py)
        input_data['Slope_Elevation_Interaction'] = slope * elevation / 1000
        input_data['Slope_Curvature_Interaction'] = slope * curvature
        input_data['Rainfall_Saturation'] = rainfall * soil_sat
        input_data['Rainfall_24h'] = rainfall * 0.4
        input_data['Rainfall_72h'] = rainfall * 0.9
        input_data['Rainfall_Intensity_Change'] = (input_data['Rainfall_72h'] - input_data['Rainfall_24h']) / 48
        
        # Handle LULC
        lulc = data.get('LULC', 'Forest')
        for cat in ['Agriculture', 'Barren', 'Forest', 'Grassland', 'Urban']:
            input_data[f'LULC_{cat}'] = 1.0 if lulc == cat else 0.0
            
        # Composite risk indicator (approximation of the logic in train_model.py)
        risk_score = (
            (slope / 70 * 0.35) + 
            (rainfall / 2000 * 0.25) + 
            ((1 - veg_cover) * 0.15) + 
            (soil_sat * 0.15) + 
            (abs(curvature) * 0.1)
        )
        input_data['Risk_Composite'] = risk_score

        # Create feature vector in correct order
        features = []
        for name in feature_names:
            features.append(input_data.get(name, 0.0))
            
        # Scale and predict
        features_scaled = scaler.transform([features])
        proba = model.predict_proba(features_scaled)[0][1]
        
        # Categorize risk (consistent with training logic)
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
