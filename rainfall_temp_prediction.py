#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("weather_dataset.csv")

# Feature Selection
X = df[["humidity", "wind_speed", "weather_condition"]]
y_rainfall = df["rainfall"]
y_temp = df["temperature"]

# Encode categorical variable
label_encoder = LabelEncoder()
X["weather_condition"] = label_encoder.fit_transform(X["weather_condition"])

# Train-Test Split
X_train, X_test, y_train_rain, y_test_rain = train_test_split(X, y_rainfall, test_size=0.2, random_state=42)
X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)

# Train Models
rainfall_lr = LinearRegression()
rainfall_lr.fit(X_train, y_train_rain)

temp_lr = LinearRegression()
temp_lr.fit(X_train, y_train_temp)

rainfall_dt = DecisionTreeRegressor()
rainfall_dt.fit(X_train, y_train_rain)

temp_dt = DecisionTreeRegressor()
temp_dt.fit(X_train, y_train_temp)

rainfall_rf = RandomForestRegressor()
rainfall_rf.fit(X_train, y_train_rain)

temp_rf = RandomForestRegressor()
temp_rf.fit(X_train, y_train_temp)

# Save Models
pickle.dump(rainfall_lr, open("rainfall_lr.pkl", "wb"))
pickle.dump(temp_lr, open("temperature_lr.pkl", "wb"))
pickle.dump(rainfall_dt, open("rainfall_dt.pkl", "wb"))
pickle.dump(temp_dt, open("temperature_dt.pkl", "wb"))
pickle.dump(rainfall_rf, open("rainfall_rf.pkl", "wb"))
pickle.dump(temp_rf, open("temperature_rf.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

# Streamlit App UI
st.title("Weather Prediction App ☁️🌧️☀️")
st.write("Predict **Rainfall** and **Temperature** based on weather conditions.")

# User Inputs
city = st.text_input("Enter City Name")
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
weather_condition = st.selectbox("Weather Condition", df["weather_condition"].unique())
model_choice = st.selectbox("Choose Model", ["Linear Regression", "Decision Tree", "Random Forest"])

# Encode categorical feature
weather_encoded = label_encoder.transform([weather_condition])[0]

# Load Selected Model
if model_choice == "Linear Regression":
    rainfall_model = pickle.load(open("rainfall_lr.pkl", "rb"))
    temp_model = pickle.load(open("temperature_lr.pkl", "rb"))
elif model_choice == "Decision Tree":
    rainfall_model = pickle.load(open("rainfall_dt.pkl", "rb"))
    temp_model = pickle.load(open("temperature_dt.pkl", "rb"))
elif model_choice == "Random Forest":
    rainfall_model = pickle.load(open("rainfall_rf.pkl", "rb"))
    temp_model = pickle.load(open("temperature_rf.pkl", "rb"))

# Make Predictions
if st.button("Predict Weather"):
    input_data = np.array([[humidity, wind_speed, weather_encoded]])
    predicted_rainfall = rainfall_model.predict(input_data)[0]
    predicted_temp = temp_model.predict(input_data)[0]
    
    # Display Results
    st.success(f"Predicted Rainfall: {predicted_rainfall:.2f} mm")
    st.success(f"Predicted Temperature: {predicted_temp:.2f}°C")


# In[ ]:




