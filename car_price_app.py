import streamlit as st
import pandas as pd
import pickle

car_data = pd.read_csv("Car_Price_Data.csv")
model = pickle.load(open("crawford_price_regressor.pkl", "rb"))
transformer = pickle.load(open("crawford_transformer.pkl", "rb"))

st.title("Car Price Predictor")

make = st.selectbox("Make", car_data["Make"].unique())
model_name = st.selectbox("Model", car_data["Model"].unique())
year = st.number_input("Year", min_value=0, max_value=2025, value=2015)
engine = st.number_input("Engine Size (L)", value=2.0)
mileage = st.number_input("Mileage", value=50000)
fuel = st.selectbox("Fuel Type", car_data["Fuel Type"].unique())
trans = st.selectbox("Transmission", car_data["Transmission"].unique())

if st.button("Predict"):
    input_df = pd.DataFrame([{
        'Make': make,
        'Model': model_name,
        'Year': year,
        'Engine Size': engine,
        'Mileage': mileage,
        'Fuel Type': fuel,
        'Transmission': trans,
    }])
    transformed = transformer.transform(input_df)
    prediction = model.predict(transformed)[0]

    st.success(f"Predicted Price: ${prediction:,.2f}")
    st.info(f"First Offer: ${prediction + 2000:,.2f}")
    st.info(f"Lowest Offer: ${prediction - 2000:,.2f}")