import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# User authentication
st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")
st.title("🔧 Predictive Maintenance Dashboard")


# Load and preprocess data
data = pd.read_csv('machinery_data.csv')
data.fillna(method='ffill', inplace=True)

# Display column info
st.sidebar.header("📝 Machinery Data Column Info")
st.sidebar.write("""
- **sensor_1**: First sensor reading, indicating a specific machinery metric.
- **sensor_2**: Second sensor reading, measuring another aspect of the machinery's performance.
- **sensor_3**: Third sensor reading, which helps in anomaly detection.
- **operational_hours**: Total hours machinery has been operational.
- **RUL**: Remaining Useful Life, estimated time before maintenance.
- **maintenance**: Maintenance status (1 = Needs Maintenance, 0 = Normal).
""")

# Feature selection and normalization
features = ['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']
target_rul = 'RUL'
target_maintenance = 'maintenance'
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Split data for regression and classification
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    data[features], data[target_rul], test_size=0.2, random_state=42
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    data[features], data[target_maintenance], test_size=0.2, random_state=42
)

# Train models
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(data[features])

# Prediction function
def predict_maintenance(features):
    rul_pred = reg_model.predict([features])
    maint_pred = clf_model.predict([features])
    cluster_pred = kmeans.predict([features])
    return {
        'RUL Prediction': rul_pred[0],
        'Maintenance Prediction': 'Needs Maintenance' if maint_pred[0] == 1 else 'Normal',
        'Anomaly Detection': 'Anomaly' if cluster_pred[0] == 1 else 'Normal'
    }

# Section for Historical Data
st.header("📂 Machinery Data")
st.write(data.head(10))

# Section for Input Data
st.header("🔧 Input Features")
st.markdown("Use the sliders to input the sensor readings and operational hours or generate random values.")
if 'generated_values' not in st.session_state:
    st.session_state['generated_values'] = None

if st.button('Generate Random Values'):
    sensor_1 = np.random.uniform(data['sensor_1'].min(), data['sensor_1'].max())
    sensor_2 = np.random.uniform(data['sensor_2'].min(), data['sensor_2'].max())
    sensor_3 = np.random.uniform(data['sensor_3'].min(), data['sensor_3'].max())
    operational_hours = np.random.uniform(data['operational_hours'].min(), data['operational_hours'].max())
    st.session_state['generated_values'] = [sensor_1, sensor_2, sensor_3, operational_hours]
    st.success("Random values generated successfully!")

if st.session_state['generated_values'] is not None:
    st.write("**Generated Values:**")
    st.write(f"Sensor 1: {st.session_state['generated_values'][0]:.2f}")
    st.write(f"Sensor 2: {st.session_state['generated_values'][1]:.2f}")
    st.write(f"Sensor 3: {st.session_state['generated_values'][2]:.2f}")
    st.write(f"Operational Hours: {st.session_state['generated_values'][3]:.2f}")

    if st.button('Use Generated Values'):
        st.session_state['input_features'] = st.session_state['generated_values']
        st.success("Generated values have been used. Navigate to the Results section to see the predictions.")

st.markdown("**Or manually input values:**")
sensor_1 = st.slider('Sensor 1', float(data['sensor_1'].min()), float(data['sensor_1'].max()), float(data['sensor_1'].mean()))
sensor_2 = st.slider('Sensor 2', float(data['sensor_2'].min()), float(data['sensor_2'].max()), float(data['sensor_2'].mean()))
sensor_3 = st.slider('Sensor 3', float(data['sensor_3'].min()), float(data['sensor_3'].max()), float(data['sensor_3'].mean()))
operational_hours = st.slider('Operational Hours', int(data['operational_hours'].min()), int(data['operational_hours'].max()), int(data['operational_hours'].mean()))

if st.button('Submit'):
    st.session_state['input_features'] = [sensor_1, sensor_2, sensor_3, operational_hours]
    st.success("Input data submitted successfully! Navigate to the Results section to see the predictions.")

# Section for Prediction Results
st.header("📊 Prediction Results")
if 'input_features' not in st.session_state:
    st.warning("Please input data first.")
else:
    input_features = st.session_state['input_features']
    prediction = predict_maintenance(input_features)
    st.write(f"**Remaining Useful Life (RUL):** {prediction['RUL Prediction']:.2f} hours")
    st.write(f"**Maintenance Status:** {prediction['Maintenance Prediction']}")
    st.write(f"**Anomaly Detection:** {prediction['Anomaly Detection']}")
    if prediction['Maintenance Prediction'] == 'Needs Maintenance':
        st.error('⚠️ Maintenance is required!')
    if prediction['Anomaly Detection'] == 'Anomaly':
        st.warning('⚠️ Anomaly detected in sensor readings!')

# Section for Data Visualizations
st.header("📊 Data Visualizations")

# Revert to original histogram for sensor readings
st.subheader("Histogram of Sensor Readings")
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(data['sensor_1'], bins=30, ax=axs[0], kde=True, color="skyblue")  # Sensor 1 in sky blue
axs[0].set_title('Sensor 1')
sns.histplot(data['sensor_2'], bins=30, ax=axs[1], kde=True, color="salmon")   # Sensor 2 in salmon
axs[1].set_title('Sensor 2')
sns.histplot(data['sensor_3'], bins=30, ax=axs[2], kde=True, color="lightgreen")  # Sensor 3 in light green
axs[2].set_title('Sensor 3')
st.pyplot(fig)


# Scatter plot for sensor readings vs operational hours with different colors
st.subheader("Scatter Plot of Sensor Readings vs Operational Hours")
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(data['operational_hours'], data['sensor_1'], alpha=0.6)
axs[0].set_title('Operational Hours vs Sensor 1')
axs[1].scatter(data['operational_hours'], data['sensor_2'], alpha=0.6)
axs[1].set_title('Operational Hours vs Sensor 2')
axs[2].scatter(data['operational_hours'], data['sensor_3'], alpha=0.6)
axs[2].set_title('Operational Hours vs Sensor 3')
st.pyplot(fig)

# Line chart for RUL over time
st.subheader("Line Chart of RUL Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-', color="orange")  # RUL line in orange
ax.set_title('RUL Over Operational Hours')
ax.set_xlabel('Operational Hours')
ax.set_ylabel('RUL')
st.pyplot(fig)