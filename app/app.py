import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=":material/monitor_heart:",
    layout="wide"
)

from utils import inject_custom_css
inject_custom_css()

# Sidebar branding
st.sidebar.markdown("""
<div style="text-align:center; padding-top: 20px;">
    <h2 style='color: white; margin-bottom: 0;'>HeartCare AI</h2>
    <p style='color: #8892b0; font-size: 0.9rem;'>Prediction System</p>
</div>
<hr style="border-color: #1c2e64;">
""", unsafe_allow_html=True)

pages = {
    "Main": [
        st.Page("pages/Dashboard.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("pages/Prediction.py", title="Patient Prediction", icon=":material/stethoscope:"),
    ],
    "Analytics": [
        st.Page("pages/EDA.py", title="Data Overview", icon=":material/analytics:"),
        st.Page("pages/Model_Performance.py", title="Model Comparison", icon=":material/bar_chart:"),
    ],
    "Tools": [
        st.Page("pages/Reports.py", title="Reports", icon=":material/upload_file:"),
    ],
    "Others": [
        st.Page("pages/About.py", title="About", icon=":material/info:"),
    ]
}

pg = st.navigation(pages)

st.sidebar.markdown("""
<br>
<div style="background-color: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 8px; border: 1px solid #1c2e64;">
    <h4 style="color: #3b82f6; margin-top:0; font-size: 1rem;">🛡️ Early Prediction Saves Lives</h4>
    <p style="font-size: 0.8rem; color: #a8b2d1; margin-bottom: 0; line-height: 1.4;">This AI model predicts the possibility of heart disease based on medical attributes.</p>
</div>
""", unsafe_allow_html=True)

pg.run()