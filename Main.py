import os
import pandas as pd
import streamlit as st
from datetime import datetime

#st.set_page_config(page_title='DHV Preoperative Risk', layout='wide')
st.title("DHV AI Startup: Preoperative Risk Prediction (trained with 6,388 patient data)")
st.header("1 = Low risk, 2-3 = Moderate risk, 4 = High risk")

# Check if the "user_input" folder exists, if not, create it
if not os.path.exists("user_input"):
    os.makedirs("user_input")

# Get user input
age = st.number_input("Age", value=30, min_value=15)
sex = st.selectbox("Sex: 1 = Male, 2 = Female", ["1", "2"])
bmi = st.number_input("BMI", value=20.0, min_value=10.0)
asa = st.selectbox("ASA", ["1","2","3","4","5","6"])
emop = st.selectbox("EMOP: 0 = Elective, 1 = Emergency", ["0","1"])
preop_htn = st.selectbox("Preoperative HT: 0 = Normal, 1 = HT", ["0", "1"])
preop_dm = st.selectbox("Preoperative DM, 0 = Normal, 1 = DM", ["0", "1"])
preop_hb = st.number_input("Preoperative Hemoglobin (Hb)", value=12.0, min_value=10.0)
preop_plt = st.number_input("Preoperative Platelet", value=300000, min_value=5000)
preop_pt = st.number_input("Preoperative Prothrombin Time (PT)", value=0.0, min_value=0.0)
preop_aptt = st.number_input("Preoperative Activated Partial Thromboplastin Time (aPTT)", value=0.0, min_value=0.0)
preop_na = st.number_input("Preoperative Sodium (Na+)", value=135, min_value=100)
preop_k = st.number_input("Preoperative Potassium (K+)", value=4.0, min_value=0.5)
preop_gluc = st.number_input("Preoperative Blood sugar (BS)", value=100, min_value=10)
preop_alb = st.number_input("Preoperative Albumin", value=3.0, min_value=0.0)
preop_ast = st.number_input("Preoperative LFT AST", value=0.0, min_value=0.0)
preop_alt = st.number_input("Preoperative LFT ALT", value=0.0, min_value=0.0)
preop_bun = st.number_input("Preoperative BUN", value=0.0, min_value=0.0)
preop_cr = st.number_input("Preoperative Creatinine", value=0.0, min_value=0.0)

# Convert 0 or 0.00 input values to empty data
preop_pt = None if preop_pt == 0.0 else preop_pt
preop_aptt = None if preop_aptt == 0.0 else preop_aptt
preop_ast = None if preop_ast == 0.0 else preop_ast
preop_alt = None if preop_alt == 0.0 else preop_alt
preop_bun = None if preop_bun == 0.0 else preop_bun
preop_cr = None if preop_cr == 0.0 else preop_cr


# Store user input in a list
user_input = [
    {
        "date": datetime.now(),
        "age": age,
        "sex": sex,
        "bmi" : bmi,
        "asa": asa,
        "emop": emop,
        "preop_htn": preop_htn,
        "preop_dm": preop_dm,
        "preop_hb": preop_hb,
        "preop_plt": preop_plt,
        "preop_pt": preop_pt,
        "preop_aptt": preop_aptt,
        "preop_na": preop_na,
        "preop_k": preop_k,
        "preop_gluc": preop_gluc,
        "preop_alb": preop_alb,
        "preop_ast": preop_ast,
        "preop_alt": preop_alt,
        "preop_bun": preop_bun,
        "preop_cr": preop_cr
    }
]

# Display Save button
if st.button("Save"):
    # Create a DataFrame with all user input data
    df = pd.DataFrame(user_input)

    # Generate file name based on current date and time
    file_name = datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"
    file_path = os.path.join("user_input", file_name)

    # Save user input data to a single CSV file
    df.to_csv(file_path, index=False)

    # Display success message
    st.success(f"All user input data saved to {file_path}")