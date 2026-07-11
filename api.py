import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Medical Insurance Cost Predictor",
    description="An API and Web UI to predict medical insurance costs using a trained XGBoost model."
)

# Load the trained model, columns, and scaler
MODEL_PATH = "medical_insurance_xgboost.pkl"
SCALER_PATH = "medical_insurance_scaler.pkl"
COLUMNS_PATH = "medical_insurance_columns.pkl"

try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError("One or more required pickle files (.pkl) are missing.")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features_list = joblib.load(COLUMNS_PATH)
    print("Model, scaler, and columns loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    model, scaler, features_list = None, None, None

# Define Pydantic request model
class InsuranceInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: str = Field(..., description="Sex: 'male' or 'female'")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index (BMI)")
    children: int = Field(..., ge=0, le=10, description="Number of children/dependents")
    smoker: str = Field(..., description="Smoker: 'yes' or 'no'")
    region: str = Field(..., description="Region: 'northeast', 'northwest', 'southeast', 'southwest'")

@app.get("/health")
def health_check():
    """Health check endpoint to ensure model files are loaded correctly."""
    if model is None or scaler is None or features_list is None:
        raise HTTPException(status_code=500, detail="Models or configuration not loaded correctly.")
    return {"status": "healthy", "features": features_list}

@app.post("/predict")
def predict(data: InsuranceInput):
    """Predict the medical insurance cost based on input parameters."""
    if model is None or scaler is None or features_list is None:
        raise HTTPException(status_code=500, detail="Model and scaler not initialized.")
    
    try:
        # Preprocess input data
        # Mapping Sex
        sex_str = data.sex.lower().strip()
        if sex_str not in ["male", "female"]:
            raise HTTPException(status_code=400, detail="Sex must be 'male' or 'female'.")
        sex_val = 1 if sex_str == "male" else 0
        
        # Mapping Smoker
        smoker_str = data.smoker.lower().strip()
        if smoker_str not in ["yes", "no"]:
            raise HTTPException(status_code=400, detail="Smoker must be 'yes' or 'no'.")
        smoker_val = 1 if smoker_str == "yes" else 0
        
        # Mapping Region (one-hot encoding)
        region_str = data.region.lower().strip()
        regions = ["northeast", "northwest", "southeast", "southwest"]
        if region_str not in regions:
            raise HTTPException(status_code=400, detail="Region must be one of: northeast, northwest, southeast, southwest.")
        
        region_northeast = 1 if region_str == "northeast" else 0
        region_northwest = 1 if region_str == "northwest" else 0
        region_southeast = 1 if region_str == "southeast" else 0
        region_southwest = 1 if region_str == "southwest" else 0
        
        # Scale BMI
        # StandardScaler was fit on a 2D array, so we must reshape the input accordingly
        bmi_scaled = float(scaler.transform([[data.bmi]])[0][0])
        
        # Construct prediction row mapping exactly to trained columns
        # Column names: ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
        input_dict = {
            'age': data.age,
            'sex': sex_val,
            'bmi': bmi_scaled,
            'children': data.children,
            'smoker': smoker_val,
            'region_northeast': region_northeast,
            'region_northwest': region_northwest,
            'region_southeast': region_southeast,
            'region_southwest': region_southwest
        }
        
        # Build DataFrame with columns in exact order
        df_input = pd.DataFrame([input_dict])
        df_input = df_input[features_list]
        
        # Predict Log Charges
        pred_log = float(model.predict(df_input)[0])
        
        # Invert Log1p scaling: expm1(pred_log)
        predicted_charges = float(np.expm1(pred_log))
        
        return {
            "prediction": round(predicted_charges, 2),
            "currency": "USD"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def index():
    """Beautiful web dashboard to interact with the medical prediction model."""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index HTML not found.")
