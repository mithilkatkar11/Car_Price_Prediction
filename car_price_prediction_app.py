import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle

# Load the pickled model
model_file = open('car_price_prediction_model.pkl', 'rb')
model = pickle.load(model_file)

def predict_price(year, present_price, kms_driven, owner, fuel_type, seller_type, transmission):
    # Calculate the number of years
    current_year = datetime.datetime.now().year
    no_of_years = current_year - year
    
    # Convert categorical variables to one-hot encoding
    fuel_type_diesel = 1 if fuel_type == 'Diesel' else 0
    fuel_type_petrol = 1 if fuel_type == 'Petrol' else 0
    seller_type_individual = 1 if seller_type == 'Individual' else 0
    transmission_manual = 1 if transmission == 'Manual' else 0
    
    # Make prediction
    prediction_data = [[present_price, kms_driven, owner, no_of_years, fuel_type_diesel, fuel_type_petrol, 
                        seller_type_individual, transmission_manual]]
    prediction = model.predict(prediction_data)
    return prediction[0]

# Streamlit UI
st.title('Car Price Prediction')

year = st.slider('Year of purchase', min_value=2000, max_value=datetime.datetime.now().year, step=1)
present_price = st.number_input('Present Price (in lakhs)', min_value=0.1, step=0.1)
kms_driven = st.number_input('Kilometers driven', min_value=0, step=1000)
owner = st.number_input('Number of previous owners', min_value=0, step=1)
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])

if st.button('Predict'):
    predicted_price = predict_price(year, present_price, kms_driven, owner, fuel_type, seller_type, transmission)
    st.success(f'Predicted Selling Price: {predicted_price:.2f} lakhs')
