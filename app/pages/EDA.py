import streamlit as st
import pandas as pd
import os
import altair as alt


st.title("Exploratory Data Analysis", anchor=False)
st.markdown("Understand the patterns and relationships in heart disease data", help="This dashboard provides a comprehensive view of the patient data used to train the model.")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, 'data', 'heart.csv')
    try:
        return pd.read_csv(data_path)
    except Exception:
        return None

df = load_data()

if df is not None:
    # Top level metrics
    total_patients = len(df)
    disease_yes = len(df[df['target'] == 1])
    disease_no = len(df[df['target'] == 0])
    total_features = df.shape[1] - 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", f"{total_patients:,}", border=True)
    with col2:
        st.metric("Heart Disease (Yes)", f"{disease_yes:,}", f"{(disease_yes/total_patients)*100:.2f}%", border=True)
    with col3:
        st.metric("No Heart Disease", f"{disease_no:,}", f"{(disease_no/total_patients)*100:.2f}%", border=True)
    with col4:
        st.metric("Total Features", f"{total_features}", border=True)

    st.write("")

    # Row 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("**Age Distribution** :material/info:")
            chart = alt.Chart(df).mark_bar(opacity=0.8, color="#0A66C2").encode(
                x=alt.X('age:Q', bin=alt.Bin(maxbins=20), title='Age'),
                y=alt.Y('count()', title='Count'),
                tooltip=['count()']
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("**Heart Disease by Gender** :material/info:")
            # 1 = Male, 0 = Female
            gender_df = df.copy()
            gender_df['Gender'] = gender_df['sex'].map({1: 'Male', 0: 'Female'})
            gender_counts = gender_df['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            chart = alt.Chart(gender_counts).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Gender", type="nominal", scale=alt.Scale(range=["#0A66C2", "#EF4444"])),
                tooltip=['Gender', 'Count']
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

    with col3:
        with st.container(border=True):
            st.markdown("**Correlation Heatmap** :material/info:")
            corr = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']].corr().reset_index().melt('index')
            chart = alt.Chart(corr).mark_rect().encode(
                x=alt.X('index:O', title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('variable:O', title=None),
                color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                tooltip=['index', 'variable', 'value']
            ).properties(height=280)
            
            # Add text labels
            text = chart.mark_text(baseline='middle').encode(
                text=alt.Text('value:Q', format='.2f'),
                color=alt.condition(
                    abs(alt.datum.value) > 0.4,
                    alt.value('white'),
                    alt.value('black')
                )
            )
            st.altair_chart(chart + text, use_container_width=True)

    # Row 2
    col4, col5, col6 = st.columns(3)
    
    with col4:
        with st.container(border=True):
            st.markdown("**Chest Pain Type Analysis** :material/info:")
            cp_df = df.copy()
            cp_df['Chest Pain'] = cp_df['cp'].map({0: 'TA', 1: 'ATA', 2: 'NAP', 3: 'ASY'})
            chart = alt.Chart(cp_df).mark_bar().encode(
                x=alt.X('Chest Pain:N', title='Chest Pain Type', sort=['TA', 'ATA', 'NAP', 'ASY'], axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count()', title='Count'),
                color=alt.Color('Chest Pain:N', legend=None, scale=alt.Scale(range=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]))
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
            
    with col5:
        with st.container(border=True):
            st.markdown("**Cholesterol Distribution** :material/info:")
            chart = alt.Chart(df).mark_bar(opacity=0.8, color="#10B981").encode(
                x=alt.X('chol:Q', bin=alt.Bin(maxbins=20), title='Cholesterol (mg/dl)'),
                y=alt.Y('count()', title='Count'),
                tooltip=['count()']
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
            
    with col6:
        with st.container(border=True):
            st.markdown("**Blood Pressure Distribution** :material/info:")
            chart = alt.Chart(df).mark_bar(opacity=0.8, color="#6366F1").encode(
                x=alt.X('trestbps:Q', bin=alt.Bin(maxbins=20), title='Resting Blood Pressure (mm Hg)'),
                y=alt.Y('count()', title='Count'),
                tooltip=['count()']
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

    st.caption(":bulb: These insights help us understand the data better and build a more accurate heart disease prediction model.")
            
else:
    st.error("Could not locate `data/heart.csv`. Please ensure it is placed in the correct folder.", icon=":material/error:")