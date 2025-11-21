import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("🏠 Housing Price Prediction App")
st.write("Enter the house details below:")

# Input fields
area = st.number_input("Area (sq ft)", min_value=100, max_value=20000)
bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10)
bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=10)
stories = st.number_input("Number of stories", min_value=1, max_value=5)

mainroad = st.selectbox("Near Main Road?", ["yes", "no"])
guestroom = st.selectbox("Guest Room?", ["yes", "no"])
basement = st.selectbox("Basement?", ["yes", "no"])
hotwater = st.selectbox("Hot Water Heating?", ["yes", "no"])
aircon = st.selectbox("Air Conditioning?", ["yes", "no"])
prefarea = st.selectbox("Preferred Area?", ["yes", "no"])
furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
parking = st.number_input("Parking spaces", min_value=0, max_value=5)

# -------------------------------------------------------
# ✅ Define encode BEFORE using it
# -------------------------------------------------------
def encode_yes_no(value):
    return 1 if value == "yes" else 0
# -------------------------------------------------------

# Build DataFrame EXACTLY in model order
data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [encode_yes_no(mainroad)],
    "guestroom": [encode_yes_no(guestroom)],
    "basement": [encode_yes_no(basement)],
    "hotwaterheating": [encode_yes_no(hotwater)],
    "airconditioning": [encode_yes_no(aircon)],
    "parking": [parking],                      # correct order
    "prefarea": [encode_yes_no(prefarea)],     # correct order
    "furnishingstatus": [
        0 if furnishing == "unfurnished" else 
        1 if furnishing == "semi-furnished" else 
        2
    ]
})

if st.button("Predict Price"):
    prediction = model.predict(data)[0]
    st.success(f"Estimated House Price: ₦{prediction:,.0f}")
