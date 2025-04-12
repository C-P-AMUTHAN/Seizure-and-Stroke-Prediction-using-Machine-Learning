from flask import Flask, request, jsonify, render_template
import logging
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = app.logger

# Global variables for models and scalers
seizure_model = None
seizure_scaler = None
stroke_model = None
stroke_scaler = None
stroke_encoders = None

def load_models():
    """Load ML models and scalers with error handling"""
    global seizure_model, seizure_scaler, stroke_model, stroke_scaler, stroke_encoders
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.warning(f"Created models directory at {models_dir}")
        return False
        
    try:
        # Load seizure model and scaler
        seizure_model_path = os.path.join(models_dir, 'seizure_svm_model.pkl')
        seizure_scaler_path = os.path.join(models_dir, 'seizure_scaler.pkl')
        
        if os.path.exists(seizure_model_path) and os.path.exists(seizure_scaler_path):
            seizure_model = joblib.load(seizure_model_path)
            seizure_scaler = joblib.load(seizure_scaler_path)
            logger.info("Seizure model and scaler loaded successfully")
        else:
            logger.error("Seizure model or scaler not found")
            
        # Load stroke model components
        stroke_model_path = os.path.join(models_dir, 'stroke_rf_model.pkl')
        stroke_scaler_path = os.path.join(models_dir, 'stroke_scaler.pkl')
        stroke_encoders_path = os.path.join(models_dir, 'stroke_label_encoders.pkl')
        
        if all(os.path.exists(p) for p in [stroke_model_path, stroke_scaler_path, stroke_encoders_path]):
            stroke_model = joblib.load(stroke_model_path)
            stroke_scaler = joblib.load(stroke_scaler_path)
            stroke_encoders = joblib.load(stroke_encoders_path)
            logger.info("Stroke model components loaded successfully")
        else:
            logger.error("One or more stroke model components not found")
            
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def validate_seizure_input(data):
    """Validate seizure prediction input data"""
    required_fields = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'Patient_Age', 'Medication_Use']
    
    # Check for missing fields
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Convert and validate numeric fields
    try:
        cleaned_data = {
            'EEG1': float(data['EEG1']),
            'EEG2': float(data['EEG2']),
            'EEG3': float(data['EEG3']),
            'EEG4': float(data['EEG4']),
            'Patient_Age': float(data['Patient_Age']),
            'Medication_Use': int(data['Medication_Use'])
        }
        
        # Validate ranges
        if not (0 <= cleaned_data['Patient_Age'] <= 120):
            raise ValueError("Age must be between 0 and 120")
        if cleaned_data['Medication_Use'] not in [0, 1]:
            raise ValueError("Medication_Use must be 0 or 1")
            
        return cleaned_data
    except ValueError as e:
        raise ValueError(f"Invalid numeric value: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data
        input_data = request.get_json()
        if not input_data:
            logger.error("No input data provided")
            return jsonify({"error": "No input data provided"}), 400
            
        logger.info(f"Received prediction request with data: {input_data}")
        
        # Check model type
        model_type = input_data.get('model_type', '').lower()
        if model_type not in ['seizure', 'stroke']:
            logger.error("Invalid model_type. Must be 'seizure' or 'stroke'")
            return jsonify({"error": "Invalid model_type. Must be 'seizure' or 'stroke'"}), 400
            
        if model_type == 'seizure':
            return handle_seizure_prediction(input_data)
        else:
            return handle_stroke_prediction(input_data)
            
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def handle_seizure_prediction(input_data):
    """Handle seizure prediction requests"""
    try:
        # Validate input
        try:
            cleaned_data = validate_seizure_input(input_data)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e)}), 400
            
        # Check if model and scaler are loaded
        if seizure_model is None or seizure_scaler is None:
            if not load_models():
                logger.error("Seizure model or scaler not available")
                return jsonify({"error": "Seizure model or scaler not available"}), 503
                
        # Prepare features
        try:
            # Create feature array
            features = np.array([
                cleaned_data['EEG1'],
                cleaned_data['EEG2'],
                cleaned_data['EEG3'],
                cleaned_data['EEG4'],
                cleaned_data['Patient_Age'],
                cleaned_data['Medication_Use']
            ]).reshape(1, -1)
            
            # Scale features
            if features.shape[1] == seizure_scaler.n_features_in_:
                features_scaled = seizure_scaler.transform(features)
            else:
                error_msg = f"Feature mismatch: expected {seizure_scaler.n_features_in_} features, got {features.shape[1]}"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 400
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return jsonify({"error": "Error preparing features"}), 500
            
        # Make prediction
        try:
            prediction = bool(seizure_model.predict(features_scaled)[0])
            probability = float(seizure_model.predict_proba(features_scaled)[0][1])
            
            result = {
                'seizure_prediction': prediction,
                'seizure_probability': probability,
                'status': 'success'
            }
            
            logger.info(f"Seizure prediction result: {result}")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({"error": f"Error making prediction: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in seizure prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

def handle_stroke_prediction(input_data):
    """Handle stroke prediction requests"""
    return jsonify({"error": "Stroke prediction not implemented"}), 501

if __name__ == '__main__':
    # Load models
    if not load_models():
        logger.warning("Starting server without models. Predictions will not work until models are loaded.")
        
    # Start the Flask app
    app.run(debug=True)