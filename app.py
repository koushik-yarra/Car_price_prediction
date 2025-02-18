import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model & scaler
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
sc = pickle.load(open('scaler.sav', 'rb'))
label_encoder = pickle.load(open('label_encoder.sav', 'rb'))  # Load label encoder

# Streamlit app
st.title("🚗 Used Car Price Prediction App")

# Sidebar Input Features
st.sidebar.header("Input Features")

# 1️⃣ **Car Name Input**
car_name = st.sidebar.text_input("Car Name", "Hyundai i20")  # Default value

# Encode Car Name
car_name_encoded = label_encoder.transform([car_name])[0] if car_name in label_encoder.classes_ else 0

# 2️⃣ **Other Car Features**
year = st.sidebar.slider("Year", 2003, 2020, 2015)
present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.1, value=5.59)
kms_driven = st.sidebar.number_input("Kms Driven", min_value=0, value=27000)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.sidebar.selectbox("Owner", [0, 1, 2, 3])

# 3️⃣ **Encoding Categorical Features**
fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_type_mapping = {'Dealer': 0, 'Individual': 1}
transmission_mapping = {'Manual': 0, 'Automatic': 1}

# 4️⃣ **Prepare Data for Prediction**
new_data = pd.DataFrame({
    'Car_Name': [car_name_encoded],  # Encoded Car Name
    'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Fuel_Type': [fuel_type_mapping[fuel_type]],
    'Seller_Type': [seller_type_mapping[seller_type]],
    'Transmission': [transmission_mapping[transmission]],
    'Owner': [owner]
})

# 5️⃣ **Preprocess the input data**
scaled_data = sc.transform(new_data)

# 6️⃣ **Predict Button**
if st.button("Predict Price"):
    prediction = loaded_model.predict(scaled_data)
    st.success(f"💰 Predicted Selling Price: ₹ {round(prediction[0], 2)} Lakhs")

# 7️⃣ **Show Entered Car Name**
st.sidebar.write(f"**Car Selected:** {car_name}")
