import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

# Custom Styling Injection
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.15) 0%, transparent 40%),
                    #0b0f19;
        color: #f3f4f6;
    }
    
    /* Title style */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Result card styling */
    .result-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeIn 0.4s ease-out;
    }
    
    .result-title {
        color: #9ca3af;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #10b981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Load Resources
MODEL_PATH = "medical_insurance_xgboost.pkl"
SCALER_PATH = "medical_insurance_scaler.pkl"
COLUMNS_PATH = "medical_insurance_columns.pkl"

@st.cache_resource
def load_ml_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(COLUMNS_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(COLUMNS_PATH)
    return model, scaler, features

model, scaler, features_list = load_ml_resources()

# Render Header
st.markdown('<div class="main-title">Insurance Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict annual medical insurance cost using XGBoost machine learning model</div>', unsafe_allow_html=True)

# Check if model files loaded properly
if model is None or scaler is None or features_list is None:
    st.error("Error: Could not load machine learning model files. Please verify the pickle files exist in the repository root.")
else:
    # Form Layout using Columns
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, help="Age in years")
            sex = st.selectbox("Sex", options=["Female", "Male"], index=0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f", help="Body Mass Index")
            
        with col2:
            children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1, help="Number of dependents/children")
            smoker = st.selectbox("Smoker", options=["No", "Yes"], index=0)
            region = st.selectbox("Region", options=["Northeast", "Northwest", "Southeast", "Southwest"], index=0)
        
        # Submit button
        submit_button = st.form_submit_button("Calculate Premium", use_container_width=True)
        
    if submit_button:
        try:
            # Map Categorical Variables
            sex_val = 1 if sex == "Male" else 0
            smoker_val = 1 if smoker == "Yes" else 0
            
            # Map Region (one-hot encoding)
            region_northeast = 1 if region == "Northeast" else 0
            region_northwest = 1 if region == "Northwest" else 0
            region_southeast = 1 if region == "Southeast" else 0
            region_southwest = 1 if region == "Southwest" else 0
            
            # Scale BMI using standard scaler (fit on 2D)
            bmi_scaled = float(scaler.transform([[bmi]])[0][0])
            
            # Create feature dict matching XGBoost expected column sequence
            input_dict = {
                'age': age,
                'sex': sex_val,
                'bmi': bmi_scaled,
                'children': children,
                'smoker': smoker_val,
                'region_northeast': region_northeast,
                'region_northwest': region_northwest,
                'region_southeast': region_southeast,
                'region_southwest': region_southwest
            }
            
            # Construct DataFrame and align columns
            df_input = pd.DataFrame([input_dict])
            df_input = df_input[features_list]
            
            # Make prediction (predicted output is log(charges + 1))
            pred_log = float(model.predict(df_input)[0])
            
            # Invert log transform using expm1
            predicted_charges = float(np.expm1(pred_log))
            
            # Display Prediction
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Estimated Annual Premium</div>
                <div class="result-value">${predicted_charges:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Inference Error: {e}")
