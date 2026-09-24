# Heart Disease Prediction

## Overview

HeartCare AI is a Machine Learning-powered prediction system that estimates
the possibility of heart disease based on various medical attributes. This
system is designed as an interactive web application, allowing users to input
patient data and receive predictions, explore the underlying dataset through
Exploratory Data Analysis (EDA), compare model performances, and generate
comprehensive reports. Early prediction is crucial for timely intervention
and saving lives.

## Dataset

The project utilizes the Heart Disease Dataset, which contains various medical
attributes (features) used to predict the presence of heart disease in a
patient.

**Dataset Link:**
[Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
(Link provided by user)

## Project Description

The Heart Disease Prediction application is built with Streamlit and features
a multi-page interactive dashboard.

Key features include:

- **Dashboard**: High-level overview of the application and quick statistics.
- **Patient Prediction**: An interactive interface for users to enter medical
  parameters and receive a heart disease prediction utilizing a trained
  machine learning model.
- **Data Overview (EDA)**: Interactive analytics and visualizations to explore
  the dataset's distributions and feature correlations.
- **Model Comparison**: Evaluation metrics comparing different machine learning
  models to highlight the effectiveness of the chosen predictive algorithm.
- **Reports**: Functionality to generate and download medical reports (e.g.,
  in PDF format).

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Web framework for building the interactive dashboard.
- **Scikit-Learn**: Machine learning library used for training and evaluating
  prediction models.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization libraries for EDA.
- **Joblib**: For saving and loading trained machine learning models.
- **fpdf2**: Used for generating PDF reports.

## Setup and Run Instructions

To run this project locally, follow these steps:

1. **Navigate to the project directory**:

   ```bash
   cd Heart-Disease-Prediction-main
   ```

2. **Create a Virtual Environment** (Recommended):

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the Required Dependencies**:

   Install all the necessary packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:

   Navigate to the `app` directory and start the Streamlit server:

   ```bash
   cd app
   python -m streamlit run app.py
   ```

   The application will automatically open in your default web browser
   (typically at `http://localhost:8501`).

## Key Information

- **Disclaimer**: This tool is for informational and educational purposes
  only and is not intended to serve as professional medical advice, diagnosis,
  or treatment. Always seek the advice of your physician or other qualified
  health providers with any questions you may have regarding a medical
  condition.
- **Customization**: The application includes custom CSS for an enhanced
  user experience and modern UI branding.
