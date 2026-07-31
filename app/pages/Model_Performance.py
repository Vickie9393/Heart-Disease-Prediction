import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys
import os

# Ensure the app folder is in the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_model_and_scaler


st.title("Model Comparison", anchor=False)
st.markdown("Compare the performance of different machine learning models")

# Load real model for feature importance, mock the rest for visualization
model, scaler = load_model_and_scaler()

# --- Mock Data for UI Demonstration ---
model_metrics = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'Decision Tree', 'KNN'],
    'Accuracy': [88.52, 87.21, 85.25, 84.10, 81.97, 78.69],
    'Precision': [0.89, 0.87, 0.85, 0.84, 0.82, 0.79],
    'Recall': [0.88, 0.87, 0.86, 0.84, 0.82, 0.79],
    'F1-score': [0.88, 0.87, 0.85, 0.84, 0.81, 0.78],
    'ROC-AUC': [0.94, 0.93, 0.91, 0.90, 0.87, 0.83],
    'Color': ['#8B5CF6', '#EF4444', '#3B82F6', '#F59E0B', '#10B981', '#0EA5E9']
})

def plot_sparkline(data, color):
    chart = alt.Chart(pd.DataFrame({'y': data, 'x': range(len(data))})).mark_area(
        line={'color': color},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color=color, offset=0),
                   alt.GradientStop(color='white', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        ),
        opacity=0.3
    ).encode(
        x=alt.X('x:Q', axis=None),
        y=alt.Y('y:Q', axis=None, scale=alt.Scale(domain=[min(data)*0.9, max(data)*1.1]))
    ).properties(height=40)
    return chart

st.write("")

# 1. Top Model Cards
cols = st.columns(6)
for i, (index, row) in enumerate(model_metrics.iterrows()):
    with cols[i]:
        with st.container(border=True):
            if index == 0:
                st.markdown(f"**{row['Model']}** :material/workspace_premium:")
            else:
                st.markdown(f"**{row['Model']}**")
            
            st.markdown(f"<div style='text-align:center; color:{row['Color']}; font-size:24px; font-weight:bold;'>{row['Accuracy']}%</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center; color:#6B7280; font-size:12px; margin-bottom: 10px;'>Accuracy</div>", unsafe_allow_html=True)
            
            # Sub-metrics
            st.markdown(f"<div style='font-size:12px; display:flex; justify-content:space-between; color:#4B5563;'><span>Precision</span> <span>{row['Precision']:.2f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12px; display:flex; justify-content:space-between; color:#4B5563;'><span>Recall</span> <span>{row['Recall']:.2f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12px; display:flex; justify-content:space-between; color:#4B5563;'><span>F1-score</span> <span>{row['F1-score']:.2f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12px; display:flex; justify-content:space-between; color:#4B5563;'><span>ROC-AUC</span> <span>{row['ROC-AUC']:.2f}</span></div>", unsafe_allow_html=True)
            
            st.write("")
            # Sparkline mockup
            spark_data = np.random.normal(row['Accuracy'], 2, 20).cumsum()
            st.altair_chart(plot_sparkline(spark_data, row['Color']), use_container_width=True)

st.write("")

# 2. Confusion Matrix and ROC Curve
col1, col2 = st.columns([1, 1.2])

with col1:
    with st.container(border=True):
        st.markdown("**Confusion Matrix** (Best Model: Random Forest)")
        
        cm_data = pd.DataFrame([
            {'Actual': 'No Disease (0)', 'Predicted': 'No Disease (0)', 'Value': 105},
            {'Actual': 'No Disease (0)', 'Predicted': 'Disease (1)', 'Value': 15},
            {'Actual': 'Disease (1)', 'Predicted': 'No Disease (0)', 'Value': 18},
            {'Actual': 'Disease (1)', 'Predicted': 'Disease (1)', 'Value': 111}
        ])
        
        cm_chart = alt.Chart(cm_data).mark_rect().encode(
            x=alt.X('Predicted:O', title='Predicted', sort='-x'),
            y=alt.Y('Actual:O', title='Actual'),
            color=alt.Color('Value:Q', scale=alt.Scale(scheme='blues'), legend=None)
        )
        
        text = cm_chart.mark_text(baseline='middle', size=16, fontWeight='bold').encode(
            text='Value:Q',
            color=alt.condition(alt.datum.Value > 50, alt.value('white'), alt.value('black'))
        )
        
        st.altair_chart((cm_chart + text).properties(height=320), use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown("**ROC Curve Comparison**")
        
        x = np.linspace(0, 1, 100)
        roc_dfs = []
        for index, row in model_metrics.iterrows():
            auc = row['ROC-AUC']
            y = 1 - (1 - x) ** (auc * 10)
            df_temp = pd.DataFrame({'False Positive Rate': x, 'True Positive Rate': y, 'Model': f"{row['Model']} (AUC = {auc:.2f})"})
            roc_dfs.append(df_temp)
            
        roc_dfs.append(pd.DataFrame({'False Positive Rate': x, 'True Positive Rate': x, 'Model': 'Random Guess (AUC = 0.50)'}))
        roc_data = pd.concat(roc_dfs)
        
        domain = [f"{row['Model']} (AUC = {row['ROC-AUC']:.2f})" for i, row in model_metrics.iterrows()] + ['Random Guess (AUC = 0.50)']
        range_colors = model_metrics['Color'].tolist() + ['#9CA3AF']
        
        roc_chart = alt.Chart(roc_data).mark_line().encode(
            x='False Positive Rate:Q',
            y='True Positive Rate:Q',
            color=alt.Color('Model:N', scale=alt.Scale(domain=domain, range=range_colors)),
            strokeDash=alt.condition(alt.datum.Model == 'Random Guess (AUC = 0.50)', alt.value([5, 5]), alt.value([0]))
        ).properties(height=320)
        
        st.altair_chart(roc_chart, use_container_width=True)

st.write("")

# 3. Model Performance Summary and Insights
col3, col4 = st.columns([2, 1])

with col3:
    with st.container(border=True):
        st.markdown("**Model Performance Summary**")
        
        display_df = model_metrics.copy()
        display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x}%")
        
        st.dataframe(
            display_df.drop(columns=['Color']),
            hide_index=True,
            use_container_width=True
        )

with col4:
    with st.container(border=True):
        st.markdown("**Insights** :material/lightbulb:")
        st.markdown("""
        * **Random Forest** performs the best with 88.52% accuracy and 0.94 ROC-AUC.
        * **XGBoost** is the second-best performer.
        * **KNN** shows the lowest performance among all models.
        * All models perform significantly better than random guessing.
        """)