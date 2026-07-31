import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_model_and_scaler

st.set_page_config(page_title="Batch Reports", page_icon="📄")

st.title("📄 Batch Prediction Reports")
st.write("Upload a CSV file containing multiple patient records to generate bulk risk assessments.")

model, scaler = load_model_and_scaler()

uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type="csv")

if uploaded_file is not None and model and scaler:
    # Read uploaded file
    input_df = pd.read_csv(uploaded_file)
    
    expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Verify the uploaded CSV has the correct columns
    if all(col in input_df.columns for col in expected_cols):
        st.success("File uploaded successfully. Processing predictions...")
        
        # Isolate features, scale, and predict
        features = input_df[expected_cols]
        scaled_features = scaler.transform(features)
        
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[:, 1]
        
        # Append results to the dataframe
        result_df = input_df.copy()
        result_df['Predicted_Risk'] = predictions
        result_df['Risk_Probability'] = (probabilities * 100).round(2).astype(str) + "%"
        
        st.write("### Prediction Results")
        st.dataframe(result_df)
        
        # Create downloadable CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Report",
            data=csv,
            file_name='patient_risk_report.csv',
            mime='text/csv',
        )
    else:
        st.error(f"Uploaded CSV is missing required columns. Expected: {', '.join(expected_cols)}")