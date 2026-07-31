import streamlit as st
import pandas as pd
import altair as alt
import sys
import os

# Ensure the app folder is in the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_model_and_scaler


st.title("Heart Disease Prediction", anchor=False)
st.markdown("Enter patient details to predict the likelihood of heart disease.")

model, scaler = load_model_and_scaler()

if model and scaler:
    with st.form("prediction_form", border=True):
        st.markdown("**Patient Information** :material/person:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("1. Age (years)", min_value=1, max_value=120, value=54)
            chol = st.number_input("5. Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            exang = st.selectbox("9. Exercise Induced Angina", ["No", "Yes"])
            
        with col2:
            sex = st.selectbox("2. Sex", ["Female", "Male"], index=1)
            fbs = st.selectbox("6. Fasting Blood Sugar > 120", ["No", "Yes"])
            oldpeak = st.number_input("10. Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            
        with col3:
            cp_map = {"Typical Angina (TA)": 0, "Atypical Angina (ATA)": 1, "Non-anginal Pain (NAP)": 2, "Asymptomatic (ASY)": 3}
            cp_choice = st.selectbox("3. Chest Pain Type", list(cp_map.keys()), index=1)
            
            restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            restecg_choice = st.selectbox("7. Resting ECG", list(restecg_map.keys()))
            
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_choice = st.selectbox("11. ST Slope", list(slope_map.keys()))
            
        with col4:
            trestbps = st.number_input("4. Resting Blood Pressure", min_value=50, max_value=250, value=120)
            thalach = st.number_input("8. Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
            
            ca = st.selectbox("12. Major vessels (0-4)", [0, 1, 2, 3, 4])
            thal_map = {"Normal": 0, "Fixed defect": 1, "Reversible defect": 2, "Other": 3}
            thal_choice = st.selectbox("13. Thal", list(thal_map.keys()))
            
        st.write("")
        submitted = st.form_submit_button("Predict Heart Disease", type="primary", use_container_width=True)
        
        if submitted:
            # Format inputs
            sex_val = 1 if sex == "Male" else 0
            exang_val = 1 if exang == "Yes" else 0
            fbs_val = 1 if fbs == "Yes" else 0
            cp_val = cp_map[cp_choice]
            restecg_val = restecg_map[restecg_choice]
            slope_val = slope_map[slope_choice]
            thal_val = thal_map[thal_choice]
            
            input_data = pd.DataFrame([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
                                        thalach, exang_val, oldpeak, slope_val, ca, thal_val]],
                                      columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            
            # Predict
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)[0][1]
            
            st.session_state['prediction_run'] = True
            st.session_state['pred_class'] = prediction[0]
            st.session_state['pred_prob'] = probability
            
            # Store inputs for displaying cards
            st.session_state['inputs'] = {
                "Age": f"{age} years",
                "Sex": sex,
                "Chest Pain Type": cp_choice.split("(")[0].strip(),
                "Resting BP": f"{trestbps} mm Hg",
                "Cholesterol": f"{chol} mg/dl",
                "Fasting BS": fbs,
                "Resting ECG": restecg_choice,
                "Max Heart Rate": f"{thalach} bpm",
                "Exercise Angina": exang,
                "Oldpeak": str(oldpeak),
                "ST Slope": slope_choice
            }

if st.session_state.get('prediction_run', False):
    st.write("---")
    st.markdown("### Prediction Result")
    
    prob_disease = st.session_state['pred_prob'] * 100
    prob_no_disease = 100 - prob_disease
    is_high_risk = st.session_state['pred_class'] == 1
    
    if is_high_risk:
        banner_color = "rgba(239, 68, 68, 0.1)" # Red
        border_color = "#EF4444"
        text_color = "#EF4444"
        title = "High Risk of Heart Disease"
        subtitle = "The model predicts that the patient is likely to have heart disease."
    else:
        banner_color = "rgba(16, 185, 129, 0.1)" # Green
        border_color = "#10B981"
        text_color = "#10B981"
        title = "No Heart Disease"
        subtitle = "The model predicts that the patient is not likely to have heart disease."
        
    st.markdown(f"""
    <div style="background-color: {banner_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: {text_color}; margin: 0;">{title}</h2>
        <p style="color: #4B5563; margin: 5px 0 0 0;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        with st.container(border=True):
            st.markdown("**Patient Risk Score**")
            
            # Donut chart
            score_data = pd.DataFrame({
                'category': ['Risk', 'Safe'],
                'value': [prob_disease, prob_no_disease]
            })
            
            gauge = alt.Chart(score_data).mark_arc(innerRadius=70, cornerRadius=5).encode(
                theta=alt.Theta(field="value", type="quantitative"),
                color=alt.Color(field="category", type="nominal",
                                scale=alt.Scale(domain=['Risk', 'Safe'], range=["#EF4444", "#E5E7EB"]),
                                legend=None),
                order=alt.Order(field="category", sort="ascending")
            ).properties(height=250)
            
            text = alt.Chart(pd.DataFrame({'text': [f"{prob_disease:.1f}%"]})).mark_text(
                size=36, fontWeight='bold', color=text_color
            ).encode(text='text:N')
            
            st.altair_chart(gauge + text, use_container_width=True)
            
    with res_col2:
        with st.container(border=True):
            st.markdown("**Prediction Probabilities**")
            st.write("")
            
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:5px;'><span>Probability (Disease)</span><span style='font-weight:bold; color:#EF4444;'>{prob_disease:.2f}%</span></div>", unsafe_allow_html=True)
            st.progress(prob_disease / 100)
            
            st.write("")
            st.write("")
            
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:5px;'><span>Probability (No Disease)</span><span style='font-weight:bold; color:#10B981;'>{prob_no_disease:.2f}%</span></div>", unsafe_allow_html=True)
            st.progress(prob_no_disease / 100)
            
            st.write("")
            st.write("")
            st.markdown(f"""
            <div style="background-color: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; display: inline-block; border: 1px solid rgba(255,255,255,0.1);">
                <span style="color: #94A3B8; font-size: 14px;">Model Used: <b style="color: #F8FAFC;">Random Forest (Accuracy: 88.52%)</b></span>
            </div>
            """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Patient Medical Parameters** :material/medical_information:")
        inputs = st.session_state['inputs']
        
        keys = list(inputs.keys())
        for i in range(0, len(keys), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(keys):
                    key = keys[i+j]
                    val = inputs[key]
                    with cols[j]:
                        with st.container(border=True):
                            st.metric(label=key, value=val)

    st.write("---")
    st.markdown("### Export Report")
    st.markdown("Download a summary of the patient's medical parameters and the prediction results.")
    
    import tempfile
    
    try:
        from fpdf import FPDF
        
        class PDF(FPDF):
            def header(self):
                self.set_font('helvetica', 'B', 15)
                self.cell(0, 10, 'Heart Disease Prediction Report', border=False, align='C')
                self.ln(20)
                
            def footer(self):
                self.set_y(-15)
                self.set_font('helvetica', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        
        # Patient Parameters
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(0, 10, "Patient Medical Parameters:", ln=True)
        pdf.set_font("helvetica", size=11)
        
        for key, val in inputs.items():
            pdf.cell(80, 8, txt=str(key), border=1)
            pdf.cell(80, 8, txt=str(val), border=1, ln=True)
            
        pdf.ln(10)
        
        # Prediction Results
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(0, 10, "Prediction Results:", ln=True)
        pdf.set_font("helvetica", size=11)
        
        pdf.cell(80, 8, txt="Prediction", border=1)
        pdf.cell(80, 8, txt=str(title), border=1, ln=True)
        
        pdf.cell(80, 8, txt="Probability (Disease)", border=1)
        pdf.cell(80, 8, txt=f"{prob_disease:.2f}%", border=1, ln=True)
        
        pdf.cell(80, 8, txt="Probability (No Disease)", border=1)
        pdf.cell(80, 8, txt=f"{prob_no_disease:.2f}%", border=1, ln=True)
        
        # Create temporary file to save PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, "rb") as f:
                pdf_bytes = f.read()
                
        st.download_button(
            label="Download Report (PDF)",
            data=pdf_bytes,
            file_name="heart_disease_prediction_report.pdf",
            mime="application/pdf",
            type="primary"
        )
    except ImportError:
        st.error("Please install fpdf2 to enable PDF export (pip install fpdf2).")