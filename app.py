import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Predictor")

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="meghz0110/wellness-tourism-model",
    filename="rf_model.pkl"
)

model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction")

st.write("Enter customer details to predict purchase likelihood")

input_data = {}

for feature in model.feature_names_in_:
    input_data[feature] = st.number_input(feature)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Customer is likely to PURCHASE the package")
    else:
        st.warning("Customer is NOT likely to purchase the package")
