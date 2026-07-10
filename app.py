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
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Insurance cost Predictor</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --secondary: #06b6d4;
                --background: #0b0f19;
                --card-bg: rgba(22, 28, 45, 0.4);
                --card-border: rgba(255, 255, 255, 0.08);
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Outfit', sans-serif;
            }

            body {
                background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                            radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.15) 0%, transparent 40%),
                            var(--background);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding: 2rem 1rem;
                overflow-x: hidden;
            }

            .container {
                width: 100%;
                max-width: 600px;
                background: var(--card-bg);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 2.5rem;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            .container:hover {
                box-shadow: 0 25px 50px rgba(99, 102, 241, 0.1);
            }

            .header {
                text-align: center;
                margin-bottom: 2rem;
            }

            .header h1 {
                font-size: 2.25rem;
                font-weight: 800;
                background: linear-gradient(135deg, var(--text-main) 30%, #a5b4fc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
                letter-spacing: -0.025em;
            }

            .header p {
                color: var(--text-muted);
                font-size: 1rem;
            }

            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-bottom: 2rem;
            }

            @media (max-width: 480px) {
                .form-grid {
                    grid-template-columns: 1fr;
                }
            }

            .input-group {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            .input-group.full-width {
                grid-column: span 2;
            }

            @media (max-width: 480px) {
                .input-group.full-width {
                    grid-column: span 1;
                }
            }

            label {
                font-size: 0.875rem;
                font-weight: 500;
                color: #e5e7eb;
                letter-spacing: 0.025em;
            }

            input, select {
                background: rgba(17, 24, 39, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 0.75rem 1rem;
                color: #ffffff;
                font-size: 1rem;
                transition: all 0.2s ease;
                outline: none;
            }

            input:focus, select:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
                background: rgba(17, 24, 39, 0.8);
            }

            select option {
                background: var(--background);
                color: #ffffff;
            }

            .btn {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: #ffffff;
                border: none;
                border-radius: 12px;
                padding: 1rem;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
                width: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 0.5rem;
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
            }

            .btn:active {
                transform: translateY(0);
            }

            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .result-card {
                margin-top: 2rem;
                background: rgba(99, 102, 241, 0.06);
                border: 1px dashed rgba(99, 102, 241, 0.3);
                border-radius: 16px;
                padding: 1.5rem;
                text-align: center;
                display: none;
                animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .result-card h3 {
                color: var(--text-muted);
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }

            .result-value {
                font-size: 2.5rem;
                font-weight: 800;
                color: #10b981;
                text-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
            }

            .error-card {
                margin-top: 2rem;
                background: rgba(239, 68, 68, 0.06);
                border: 1px solid rgba(239, 68, 68, 0.2);
                border-radius: 16px;
                padding: 1rem;
                text-align: center;
                color: #f87171;
                display: none;
                font-size: 0.95rem;
            }

            .footer {
                margin-top: 2rem;
                font-size: 0.8rem;
                color: var(--text-muted);
                text-align: center;
            }

            /* Spinner Styles */
            .spinner {
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top: 3px solid #fff;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                display: none;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Insurance Estimator</h1>
                <p>Enter details below to predict annual medical insurance cost</p>
            </div>

            <form id="predictorForm" onsubmit="estimateCost(event)">
                <div class="form-grid">
                    <div class="input-group">
                        <label for="age">Age</label>
                        <input type="number" id="age" name="age" required min="18" max="100" placeholder="e.g. 35">
                    </div>

                    <div class="input-group">
                        <label for="sex">Sex</label>
                        <select id="sex" name="sex" required>
                            <option value="" disabled selected>Select sex</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="bmi">BMI</label>
                        <input type="number" id="bmi" name="bmi" step="0.1" required min="10" max="60" placeholder="e.g. 24.5">
                    </div>

                    <div class="input-group">
                        <label for="children">Children</label>
                        <input type="number" id="children" name="children" required min="0" max="10" placeholder="e.g. 0">
                    </div>

                    <div class="input-group">
                        <label for="smoker">Smoker</label>
                        <select id="smoker" name="smoker" required>
                            <option value="" disabled selected>Do you smoke?</option>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="region">Region</label>
                        <select id="region" name="region" required>
                            <option value="" disabled selected>Select region</option>
                            <option value="northeast">Northeast</option>
                            <option value="northwest">Northwest</option>
                            <option value="southeast">Southeast</option>
                            <option value="southwest">Southwest</option>
                        </select>
                    </div>
                </div>

                <button type="submit" id="submitBtn" class="btn">
                    <span id="btnText">Calculate Premium</span>
                    <div id="btnSpinner" class="spinner"></div>
                </button>
            </form>

            <div class="result-card" id="resultCard">
                <h3>Estimated Annual Cost</h3>
                <div class="result-value" id="resultValue">$0.00</div>
            </div>

            <div class="error-card" id="errorCard"></div>
        </div>

        <div class="footer">
            Powered by FastAPI & XGBoost Model
        </div>

        <script>
            async function estimateCost(event) {
                event.preventDefault();
                
                const submitBtn = document.getElementById('submitBtn');
                const btnText = document.getElementById('btnText');
                const btnSpinner = document.getElementById('btnSpinner');
                const resultCard = document.getElementById('resultCard');
                const resultValue = document.getElementById('resultValue');
                const errorCard = document.getElementById('errorCard');

                // Clear previous states
                resultCard.style.display = 'none';
                errorCard.style.display = 'none';
                
                // Show loading spinner
                submitBtn.disabled = true;
                btnText.textContent = 'Calculating...';
                btnSpinner.style.display = 'block';

                const payload = {
                    age: parseInt(document.getElementById('age').value),
                    sex: document.getElementById('sex').value,
                    bmi: parseFloat(document.getElementById('bmi').value),
                    children: parseInt(document.getElementById('children').value),
                    smoker: document.getElementById('smoker').value,
                    region: document.getElementById('region').value
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to calculate prediction.');
                    }

                    // Format result as currency
                    const formatter = new Intl.NumberFormat('en-US', {
                        style: 'currency',
                        currency: data.currency || 'USD'
                    });

                    resultValue.textContent = formatter.format(data.prediction);
                    resultCard.style.display = 'block';
                } catch (err) {
                    errorCard.textContent = err.message || 'An unexpected error occurred.';
                    errorCard.style.display = 'block';
                } finally {
                    // Reset button state
                    submitBtn.disabled = false;
                    btnText.textContent = 'Calculate Premium';
                    btnSpinner.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
