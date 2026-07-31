import joblib
import os
import streamlit as st

# @st.cache_resource ensures the model is only loaded into memory once
@st.cache_resource
def load_model_and_scaler():
    # Resolve absolute paths based on the current file location
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'heart_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run the Jupyter notebook to train and export the models.")
        return None, None

def inject_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    try:
        with open(css_path, 'r') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass