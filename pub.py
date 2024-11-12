import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess the data (use your saved model or train inline)
# You may need to load a pre-trained model if the training is too complex

# Define the prediction function
def predict_maintenance(features, reg_model, clf_model_rf, clf_model_svm, isolation_forest, scaler):
    # Standardize input features
    features = scaler.transform([features])

    # Predict RUL
    rul_pred = reg_model.predict(features)

    # Predict maintenance status using both classifiers
    maint_pred_rf = clf_model_rf.predict(features)
    maint_pred_svm = clf_model_svm.predict(features)

    # Predict anomaly using Isolation Forest
    anomaly_pred = isolation_forest.predict(features)

    return {
        'RUL Prediction': rul_pred[0],
        'Random Forest Maintenance Prediction': 'Needs Maintenance' if maint_pred_rf[0] == 1 else 'Normal',
        'SVM Maintenance Prediction': 'Needs Maintenance' if maint_pred_svm[0] == 1 else 'Normal',
        'Anomaly Detection': 'Anomaly' if anomaly_pred[0] == -1 else 'Normal'
    }

# Streamlit app UI
st.title("Machinery Maintenance Prediction")
st.write("Enter sensor readings and operational hours to predict maintenance needs and detect anomalies.")

# Input fields for user data
sensor_1 = st.number_input("Sensor 1 Reading")
sensor_2 = st.number_input("Sensor 2 Reading")
sensor_3 = st.number_input("Sensor 3 Reading")
operational_hours = st.number_input("Operational Hours")

# Button to trigger prediction
if st.button("Predict"):
    sample_features = [sensor_1, sensor_2, sensor_3, operational_hours]
    prediction = predict_maintenance(sample_features, reg_model, clf_model_rf, clf_model_svm, isolation_forest, scaler)
    st.write("Predicted Maintenance Indicator:", prediction)
