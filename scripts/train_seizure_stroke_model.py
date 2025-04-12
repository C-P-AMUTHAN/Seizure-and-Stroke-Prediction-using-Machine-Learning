import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import os

# Create models directory if it doesn't exist
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, filename):
    """Helper function to save models with proper error handling"""
    try:
        filepath = os.path.join(MODEL_DIR, filename)
        joblib.dump(model, filepath)
        print(f"Saved {filename} successfully")
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}")
        raise

# ---------------- Seizure Prediction Model ---------------- #

def train_seizure_model():
    print("\nTraining Seizure Prediction Model...")
    
    try:
        # Load dataset
        seizure_data = pd.read_csv("L:\\College\\MKCE Hackathon\\Seizure_Stroke_Prediction\\data\\seizure_dataset.csv")
        
        # Drop columns with all NaN values
        seizure_data.dropna(axis=1, how='all', inplace=True)
        
        # Prepare features and target
        X = seizure_data.drop(columns=["y"], errors='ignore')
        y = seizure_data["y"]
        
        # Handle missing values and convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # Split dataset before applying SMOTE
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Train SVM model with GridSearch
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
        
        svm = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
        svm.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate model
        y_pred = svm.predict(X_test_scaled)
        print("Best parameters:", svm.best_params_)
        print("Seizure Model Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Save models
        save_model(svm.best_estimator_, "seizure_svm_model.pkl")
        save_model(scaler, "seizure_scaler.pkl")
        
    except Exception as e:
        print(f"Error in seizure model training: {str(e)}")
        raise

# ---------------- Stroke Prediction Model ---------------- #

def train_stroke_model():
    print("\nTraining Stroke Prediction Model...")
    
    try:
        # Load dataset
        stroke_data = pd.read_csv("L:\\College\\MKCE Hackathon\\Seizure_Stroke_Prediction\\data\\stroke_dataset.csv")
        
        # Handle missing values
        stroke_data["bmi"] = stroke_data["bmi"].fillna(stroke_data["bmi"].median())
        
        # Prepare categorical and numeric features
        categorical_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
        numeric_features = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            stroke_data[col] = le.fit_transform(stroke_data[col].astype(str))
            label_encoders[col] = le
        
        # Prepare features and target
        X = stroke_data[categorical_features + numeric_features]
        y = stroke_data["stroke"]
        
        # Split dataset before applying SMOTE
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Train Random Forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate model
        y_pred = rf.predict(X_test_scaled)
        print("Stroke Model Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Save models
        save_model(rf, "stroke_rf_model.pkl")
        save_model(scaler, "stroke_scaler.pkl")
        save_model(label_encoders, "stroke_label_encoders.pkl")
        
    except Exception as e:
        print(f"Error in stroke model training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_seizure_model()
        train_stroke_model()
        print("\nAll models trained and saved successfully!")
    except Exception as e:
        print(f"\nError during model training: {str(e)}")