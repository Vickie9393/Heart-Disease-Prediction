import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("ℹ️ About the Project")

st.markdown("""
### Architecture Overview
This application serves as the frontend dashboard and microservice layer for a machine learning pipeline. 

It utilizes a **Random Forest Classifier** trained on tabular clinical data to predict the presence of cardiovascular disease.

### Tech Stack
* **Data Processing & ML:** Python, Pandas, Scikit-Learn
* **Frontend UI:** Streamlit
* **Model Serialization:** Joblib

### Disclaimer
This platform is for educational and demonstrative purposes only. It is not intended for use in clinical diagnostic environments. Always consult a medical professional for health-related concerns.
""")